"""
Stage 2's brief, made runnable: a model router that classifies each
request by complexity and sends it to the cheapest model that can
actually handle it -- logging tokens, latency, and $ for every call so
the routing decision can be defended with numbers instead of vibes.

    docs/stage-02-llm-fundamentals.md -- "Build a router that classifies
    incoming requests by complexity (regex/heuristic first, cheap-model
    classifier second) and sends 'simple' ones to a fast/cheap model,
    'complex' ones to a frontier model. Log tokens, latency, and $ per
    request for both paths."

The two-stage classifier is the part worth studying. Stage 2's doc says
"regex/heuristic first, cheap-model classifier second" and that ordering
is the whole point:

  Tier 1  A heuristic (keyword + length). Costs nothing, takes
          microseconds, and confidently settles the clear-cut cases --
          which in practice is most traffic.
  Tier 2  Only for what tier 1 couldn't decide: ask the SMALL model to
          classify. This costs a real (if tiny) call, which is exactly
          why it isn't the first thing tried.

An LLM classifier in front of every request would mean paying for two
calls on every request to save money on some of them. The heuristic
exists so that the classifier runs rarely.

Local by default, so this is free to run and to experiment with:

    USE_LOCAL_MODEL=true   -> llama3.2:3b (cheap) / llama3.1:8b (frontier)
    USE_LOCAL_MODEL=false  -> claude-haiku-4-5 / claude-opus-5

The local models cost $0, which would make the cost column useless as a
teaching device -- so local runs are also priced using the published
Claude rates, clearly labelled as simulated. You watch what the routing
WOULD cost without spending anything. Real token counts either way; only
the dollar rate is borrowed.

Setup:
    ollama pull llama3.2:3b    # ~2GB, shared with every other example
    ollama pull llama3.1:8b    # ~4.9GB, shared with stage05-multimodal

Usage:
    uv run agent.py                      # the built-in 6-request workload
    uv run agent.py "your question"      # route one request
    uv run agent.py --no-route "..."     # send everything to the big
                                         # model, to compare totals
"""

import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "true").strip().lower() == "true"

# Two tiers, whichever backend is in play. The router's job is to pick
# between them; nothing else in this file cares which pair it got.
CHEAP_MODEL = os.getenv("LOCAL_MODEL", "llama3.2:3b") if USE_LOCAL_MODEL else "claude-haiku-4-5"
FRONTIER_MODEL = "llama3.1:8b" if USE_LOCAL_MODEL else "claude-opus-5"

# $ per 1M tokens (input, output). Published Claude rates -- see
# docs/stage-02-llm-fundamentals.md's cost table. Applied to local runs
# too, as clearly-labelled simulated pricing, so the cost column has
# something to show. Re-check these before quoting them anywhere real.
PRICING = {
    CHEAP_MODEL: (1.00, 5.00),      # Haiku-tier
    FRONTIER_MODEL: (5.00, 25.00),  # Opus-tier
}

# --- Tier 1: the free heuristic -------------------------------------------
# Keywords that reliably mark a request as needing real reasoning. These
# are deliberately about the SHAPE of the task ("compare", "design",
# "why"), not its subject -- subject keywords would need endless
# maintenance as topics change.
_COMPLEX_PATTERNS = re.compile(
    r"\b(compare|contrast|analyz|evaluat|design|architect|trade-?offs?|"
    r"why|explain how|implic|strateg|recommend|refactor|debug|critique)\b",
    re.IGNORECASE,
)
# The mirror image: markers of a lookup or a one-liner.
_SIMPLE_PATTERNS = re.compile(
    r"\b(what is|what's|who is|who's|when did|when is|where is|define|"
    r"convert|translate|capital of|spell)\b",
    re.IGNORECASE,
)
LONG_REQUEST_CHARS = 180   # long requests usually carry multi-part tasks


def classify_heuristic(request: str) -> str | None:
    """Tier 1. Returns "simple", "complex", or None for 'not sure --
    escalate to the model classifier'. Returning None is a feature: a
    heuristic that guesses on every input is just a bad classifier.
    """
    complex_hit = _COMPLEX_PATTERNS.search(request)
    simple_hit = _SIMPLE_PATTERNS.search(request)

    # Both kinds of marker present ("what is X and why does it matter")
    # -- genuinely ambiguous, so don't pretend otherwise.
    if complex_hit and simple_hit:
        return None
    if complex_hit:
        return "complex"
    if simple_hit:
        # ...unless it's long enough that the lookup phrasing is only
        # the opening of a much bigger ask.
        return "simple" if len(request) < LONG_REQUEST_CHARS else None
    if len(request) >= LONG_REQUEST_CHARS:
        return "complex"
    return None


# --- Tier 2: the cheap-model classifier -----------------------------------

CLASSIFIER_PROMPT = (
    "Classify the user's request as exactly one word: SIMPLE or COMPLEX.\n"
    "SIMPLE = a fact lookup, a definition, a conversion, or anything "
    "answerable in one or two sentences with no reasoning.\n"
    "COMPLEX = anything needing comparison, analysis, planning, "
    "multi-step reasoning, or a long-form answer.\n"
    "Reply with only the single word SIMPLE or COMPLEX.\n\n"
    "Request: {request}"
)


def classify_with_model(request: str, log: "CostLog") -> str:
    """Tier 2. Deliberately runs on the CHEAP model -- paying frontier
    prices to decide whether to use the frontier model would defeat the
    entire purpose.
    """
    text, _ = call_model(
        CHEAP_MODEL,
        CLASSIFIER_PROMPT.format(request=request),
        log,
        label="classifier",
        max_tokens=8,   # one word; no reason to allow more
    )
    return "complex" if "COMPLEX" in text.upper() else "simple"


# --- Instrumentation ------------------------------------------------------


@dataclass
class Call:
    label: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float

    @property
    def cost(self) -> float:
        in_rate, out_rate = PRICING.get(self.model, (0.0, 0.0))
        return (self.tokens_in * in_rate + self.tokens_out * out_rate) / 1_000_000


@dataclass
class CostLog:
    calls: list[Call] = field(default_factory=list)

    def add(self, call: Call) -> None:
        self.calls.append(call)
        print(
            f"    [{call.label}] {call.model}  "
            f"in={call.tokens_in} out={call.tokens_out}  "
            f"{call.latency_ms:.0f}ms  ${call.cost:.6f}",
            file=sys.stderr,
        )

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls)

    @property
    def total_ms(self) -> float:
        return sum(c.latency_ms for c in self.calls)


def call_model(model: str, prompt: str, log: CostLog, label: str, max_tokens: int = 512) -> tuple[str, Call]:
    """One model call, instrumented. Token counts are the real ones
    reported by the backend -- not estimates -- which is what makes the
    cost column trustworthy even when the rate is simulated.
    """
    started = time.perf_counter()

    if USE_LOCAL_MODEL:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": max_tokens},
        )
        text = response.message.content or ""
        tokens_in = response.get("prompt_eval_count", 0) or 0
        tokens_out = response.get("eval_count", 0) or 0
    else:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens

    call = Call(
        label=label,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=(time.perf_counter() - started) * 1000,
    )
    log.add(call)
    return text, call


# --- The router itself ----------------------------------------------------


def route(request: str, log: CostLog) -> tuple[str, str]:
    """Return (model, how_it_was_decided)."""
    verdict = classify_heuristic(request)
    if verdict is not None:
        return (FRONTIER_MODEL if verdict == "complex" else CHEAP_MODEL), f"heuristic:{verdict}"

    # Tier 1 abstained -- only now is a real classifier call justified.
    verdict = classify_with_model(request, log)
    return (FRONTIER_MODEL if verdict == "complex" else CHEAP_MODEL), f"classifier:{verdict}"


def handle(request: str, log: CostLog, force_frontier: bool = False) -> str:
    print(f"\n  request: {request!r}", file=sys.stderr)
    if force_frontier:
        model, how = FRONTIER_MODEL, "forced (no routing)"
    else:
        model, how = route(request, log)
    print(f"    -> {model}  ({how})", file=sys.stderr)
    answer, _ = call_model(model, request, log, label="answer")
    return answer


WORKLOAD = [
    "What is the capital of Japan?",
    "Convert 100 Fahrenheit to Celsius.",
    "Compare the tradeoffs of microservices versus a monolith.",
    "Who wrote Pride and Prejudice?",
    "Design a caching strategy for a read-heavy API and explain why each layer earns its place.",
    "What is a vector database and why would an agent need one instead of a plain database?",
]


def run_workload(force_frontier: bool = False) -> None:
    log = CostLog()
    mode = "NO ROUTING (everything -> frontier)" if force_frontier else "ROUTED"
    print(f"\n=== {mode} ===", file=sys.stderr)

    for request in WORKLOAD:
        answer = handle(request, log, force_frontier=force_frontier)
        print(f"    answer: {answer.strip()[:100]}...", file=sys.stderr)

    routed_to_cheap = sum(1 for c in log.calls if c.label == "answer" and c.model == CHEAP_MODEL)
    classifier_calls = sum(1 for c in log.calls if c.label == "classifier")

    print(f"\n--- {mode} totals over {len(WORKLOAD)} requests ---")
    print(f"  answered by cheap model : {routed_to_cheap}/{len(WORKLOAD)}")
    print(f"  tier-2 classifier calls : {classifier_calls}  (tier 1 settled the rest for free)")
    print(f"  total latency           : {log.total_ms:.0f}ms")
    print(f"  total cost              : ${log.total_cost:.6f}")
    if USE_LOCAL_MODEL:
        print("  (cost simulated at Claude rates -- real token counts, $0 actually spent)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    force = "--no-route" in argv
    if force:
        argv.remove("--no-route")

    print(
        f"  [models] cheap={CHEAP_MODEL}  frontier={FRONTIER_MODEL}"
        f"  ({'local via Ollama' if USE_LOCAL_MODEL else 'Claude API'})",
        file=sys.stderr,
    )

    if argv:
        log = CostLog()
        print(handle(" ".join(argv), log, force_frontier=force))
        print(f"\n  total: {log.total_ms:.0f}ms  ${log.total_cost:.6f}", file=sys.stderr)
    else:
        run_workload(force_frontier=force)
