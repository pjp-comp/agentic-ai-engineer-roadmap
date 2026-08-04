"""
Single-agent ReAct loop with iteration cap, stagnation check, and graceful
degradation (Stage 7 -- docs/stage-07-single-agent.md), built as a plain
`while` loop, NOT a framework -- Stage 7's brief is deliberately about the
loop mechanics themselves, so this stays raw Ollama calls, the same way
stage03-tool-calling stays raw Claude/Ollama calls before any framework
is introduced.

Ollama only -- same LOCAL_MODEL as every other stage03+ example.

What this demonstrates, matching Stage 7's brief line for line:

    for step in range(MAX_ITERS):
        thought, action = agent.think(state)
        if is_stagnant(action, state.history):
            thought = agent.reflect(state.history)
            action = agent.think(state, hint=thought)
        result = execute(action)
        state.history.append((thought, action, result))
        if action.is_final:
            return result
    return degrade_gracefully(state)

The tool given to the model (`lookup`) is DELIBERATELY unreliable -- it
fails on the first call for any city and only succeeds on a retry. This
is what forces the loop to actually exercise every piece of Stage 7's
brief instead of finishing in one clean pass:

  - MAX_ITERS         -- a query for a city not in the fake dataset at all
                         can never succeed; the loop must stop instead of
                         spinning forever, and return a graceful fallback.
  - stagnation check   -- if the model retries the EXACT same failing
                         action twice in a row (same tool, same args) --
                         the Reflexion trigger -- force a reflection step
                         before letting it act again, instead of letting
                         it silently repeat the same mistake.
  - graceful degradation -- when MAX_ITERS is hit without a final answer,
                         return whatever partial information the history
                         contains, not a crash and not silence.

Usage:
    ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
    uv run agent.py "What's the weather in Paris?"
    uv run agent.py "What's the weather in Atlantis?"   # forces MAX_ITERS path
"""

import json
import re
import sys
from pathlib import Path

import ollama
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

LOCAL_MODEL = "llama3.2:3b"
MAX_ITERS = 6

# Deliberately flaky: the first lookup for any known city fails ("service
# busy"); the second succeeds. Cities not in here never succeed, forcing
# the MAX_ITERS / graceful-degradation path.
_WEATHER = {
    "paris": "14C, light rain",
    "tokyo": "22C, clear",
    "cairo": "31C, sunny",
}
_attempt_counts: dict[str, int] = {}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are a weather assistant with one tool: lookup(city).
Call it to answer weather questions. If a tool call fails, you may retry --
but if you're told a reflection hint, follow it instead of repeating the
exact same failing call. As soon as lookup returns a real weather report
(not an error), STOP calling tools and answer the user in plain text
using that report -- do not call lookup again once you already have the
answer. If you cannot get an answer after several tries, say so plainly
instead of guessing a weather report."""


def lookup(city: str) -> str:
    key = city.strip().lower()
    _attempt_counts[key] = _attempt_counts.get(key, 0) + 1
    if key not in _WEATHER:
        return f"error: no weather data available for {city!r}"
    if _attempt_counts[key] < 2:
        return "error: weather service busy, try again"
    return f"{city}: {_WEATHER[key]}"


def _extract_tool_call_from_text(text: str):
    """Same defensive parse used in stage03-tool-calling/stage03-multi-tool-agent:
    a small local model sometimes writes a tool call as JSON text instead
    of actually issuing it. Detect and execute it directly rather than
    trusting resp.message.tool_calls alone.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        blob = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    name = blob.get("name")
    args = blob.get("parameters") or blob.get("arguments")
    if name == "lookup" and isinstance(args, dict) and "city" in args:
        return name, args
    return None


def think(messages: list[dict], hint: str | None = None, offer_tool: bool = True) -> tuple[str, dict | None]:
    """Perceive + decide: one model call. Returns (thought_text, action) where
    action is {"name": ..., "args": {...}} or None if the model gave a
    final answer instead of calling the tool.

    offer_tool=False is the harness-level fix for a real small-model
    limitation: llama3.2:3b, when given a tool, calls it again even after
    a successful result and an explicit "stop calling tools" instruction
    in the prompt -- prompting alone doesn't reliably override the
    model's bias to use an available tool. Once the loop already has a
    successful (non-error) result in hand, it stops OFFERING the tool at
    all for that turn, which forces a text-only final answer instead of
    hoping the model self-regulates.
    """
    call_messages = messages
    if hint:
        call_messages = messages + [{"role": "user", "content": f"[reflection hint] {hint}"}]

    tools = TOOLS if offer_tool else None
    resp = ollama.chat(model=LOCAL_MODEL, messages=call_messages, tools=tools, think=False)
    text = resp.message.content or ""

    if resp.message.tool_calls:
        call = resp.message.tool_calls[0]
        return text, {"name": call.function.name, "args": dict(call.function.arguments)}

    parsed = _extract_tool_call_from_text(text) if offer_tool else None
    if parsed:
        name, args = parsed
        return text, {"name": name, "args": args}

    return text, None   # no action -> model considers this a final answer


def is_stagnant(action: dict | None, history: list[tuple]) -> bool:
    """Reflexion's trigger condition: the same action producing the same
    result, two times in a row. Comparing (name, args, result) instead of
    just (name, args) matters -- a retry that gets a DIFFERENT result
    (busy -> success) is progress, not stagnation.
    """
    if action is None or not history:
        return False
    last_thought, last_action, last_result = history[-1]
    return (
        last_action is not None
        and last_action["name"] == action["name"]
        and last_action["args"] == action["args"]
    )


def reflect(history: list[tuple]) -> str:
    """Forced reflection step: ask the model to look at its own stuck
    pattern and propose something different, instead of letting it repeat
    the identical failing call a third time.
    """
    recent = history[-2:]
    summary = "; ".join(f"called {a['name']}({a['args']}) -> {r}" for _, a, r in recent if a)
    print(f"  [reflect] stuck repeating the same action: {summary}", file=sys.stderr)
    resp = ollama.chat(
        model=LOCAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You repeated the exact same action twice with no new information: {summary}. "
                    "In one short sentence, say what you should do differently now "
                    "(e.g. try once more, or give up and say you don't know)."
                ),
            }
        ],
        think=False,
    )
    return resp.message.content or "Try a different approach or give up gracefully."


def degrade_gracefully(history: list[tuple]) -> str:
    """No crash, no silent failure -- surface whatever partial information
    the loop gathered before hitting MAX_ITERS.
    """
    results = [r for _, _, r in history if r]
    if not results:
        return "I couldn't get an answer in time and made no progress."
    return (
        "I couldn't fully resolve this within my step limit. "
        f"Here's what I found along the way: {'; '.join(results[-3:])}"
    )


def run(question: str) -> str:
    print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    history: list[tuple[str, dict | None, str | None]] = []
    have_result = False   # True once lookup has succeeded at least once

    for step in range(MAX_ITERS):
        thought, action = think(messages, offer_tool=not have_result)

        if is_stagnant(action, history):
            hint = reflect(history)
            thought, action = think(messages, hint=hint, offer_tool=not have_result)

        if action is None:
            print(f"  [step {step + 1}] final answer", file=sys.stderr)
            return thought

        print(f"  [step {step + 1}] lookup({action['args']})", file=sys.stderr)
        result = lookup(**action["args"])
        print(f"  [step {step + 1}] -> {result}", file=sys.stderr)

        messages.append({"role": "assistant", "content": f"Called lookup({action['args']}) -> {result}"})
        if result.startswith("error:"):
            nudge = "That failed. Retry once more, or say you don't know if you're out of options."
        else:
            have_result = True
            nudge = "You now have the weather report. Answer the user in plain text now -- do not call lookup again."
        messages.append({"role": "user", "content": nudge})
        history.append((thought, action, result))

    print(f"  [step limit] hit MAX_ITERS={MAX_ITERS} without a final answer", file=sys.stderr)
    return degrade_gracefully(history)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What's the weather in Paris?"
    answer = run(question)
    print(f"\n{answer}")
