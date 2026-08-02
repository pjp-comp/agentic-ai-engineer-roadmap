"""
Not an agent — no tools, no loop. Just raw calls to a local Ollama model,
so you can *see* what temperature, top-p, top-k, tokenization, and
"thinking" actually do instead of only reading about them in
docs/stage-02-llm-fundamentals.md.

Every stage after Stage 2 uses these ideas (agent.py in every 01-*/02-*
example sets temperature=0 without explaining why) — this script is where
that "why" becomes something you can watch happen, before any of the
tool-calling/loop complexity in the rest of this repo gets added on top.

One model for everything in this file: qwen3:1.7b — small (~1.4GB) and,
unlike llama3.2:3b (used elsewhere in this repo), it supports "thinking"
mode, so the same model covers every subcommand below with nothing extra
to pull.

Five things this script demonstrates, each as its own subcommand:

  same-prompt   Run the identical prompt N times at ONE fixed setting, so
                you can see run-to-run variance at that setting alone.
  compare-temps Run the identical prompt once each at temperature 0, 0.5,
                and 1.0, so you see the *spread* across settings, not just
                one number in isolation.
  compare-top-p Same idea, sweeping top_p instead of temperature (with
                temperature held at a fixed non-zero value, since top_p
                has no visible effect at temperature=0 — see "Why some
                combinations look identical" below).
  tokenize      Shows a prompt's REAL token IDs (this model's actual
                tokenizer, not an approximation) and then streams
                generation one token at a time, so "a token isn't a word
                or a character" stops being an abstract claim.
  thinking      Shows this model's separate "thinking" field (its
                reasoning, generated before the final answer) alongside
                its actual final response — two genuinely different
                fields in the API response, not the same text formatted
                two ways.

Every subcommand except `thinking` explicitly passes think=False — this
model defaults to thinking mode ON even without asking for it, which would
otherwise silently add an invisible reasoning pass to every temperature/
top-p/tokenize demo and confound what you're actually measuring.

Usage:
    ollama pull qwen3:1.7b   # one-time, ~1.4GB — the only model this file needs
    uv run sampling_params.py same-prompt --temperature 0 --runs 3
    uv run sampling_params.py same-prompt --temperature 1.0 --runs 3
    uv run sampling_params.py compare-temps
    uv run sampling_params.py compare-top-p
    uv run sampling_params.py same-prompt --prompt "Name a color." --temperature 1.0 --runs 5
    uv run sampling_params.py tokenize --prompt "The quick brown fox jumps"
    uv run sampling_params.py thinking --prompt "What is 12 + 7?"
"""

import argparse
import sys
from pathlib import Path

import ollama
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

LOCAL_MODEL = "qwen3:1.7b"
DEFAULT_PROMPT = "Name one interesting fact about elephant. One sentence only."
DEFAULT_MATH_PROMPT = "What is 12 + 7?"


def ask(prompt: str, temperature: float, top_p: float, top_k: int) -> str:
    """One raw call, no tools, no loop — the whole point of this file is
    that this function is the entire "agent." Everything before Stage 3's
    tool calling is just this: send a prompt, get text back.

    think=False is explicit and load-bearing: qwen3:1.7b defaults to
    thinking mode ON, which would otherwise add an invisible reasoning
    pass before every answer here, confounding what temperature/top_p are
    actually doing to the *visible* response. See the `thinking`
    subcommand below for where that mode is deliberately turned on instead.
    """
    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "top_p": top_p, "top_k": top_k},
        think=False,
    )
    return response.message.content or "(empty response)"


def same_prompt(prompt: str, temperature: float, top_p: float, top_k: int, runs: int) -> None:
    print(f'Prompt: "{prompt}"')
    print(f"Settings: temperature={temperature}, top_p={top_p}, top_k={top_k}\n")
    for i in range(1, runs + 1):
        print(f"[run {i}] {ask(prompt, temperature, top_p, top_k)}")
    print(
        "\nAt temperature=0, every run above should be identical (or very "
        "close to it) — the model always takes the single highest-probability "
        "token. Re-run this with --temperature 1.0 and watch the same prompt "
        "produce genuinely different answers each time. That's the entire "
        "mechanism: temperature doesn't change what the model 'knows', it "
        "changes how much of the probability distribution beyond the single "
        "most-likely token gets a chance to be picked."
    )


def compare_temps(prompt: str) -> None:
    print(f'Prompt: "{prompt}"\n')
    print("Running the SAME prompt three times, changing only temperature.")
    print("Watch the wording (and confidence) shift as temperature rises:\n")
    for temp in (0.0, 0.5, 1.0):
        print(f"--- temperature={temp} ---")
        print(ask(prompt, temperature=temp, top_p=1.0, top_k=40))
        print()
    print(
        "temperature=0: deterministic, picks the single highest-probability "
        "token every time — this is what every agent.py in this repo's "
        "01-*/02-* examples sets for tool-calling, because you want the SAME "
        "input to reliably produce the SAME tool call, not creative variation "
        "in a JSON schema.\n"
        "temperature=1.0: the full learned distribution is sampled from — "
        "more varied phrasing, but also more prone to drifting off-topic or "
        "inventing detail. Neither setting is 'more correct' — they trade "
        "consistency for variety, and which one you want depends entirely on "
        "the task (tool-calling wants consistency; brainstorming wants variety)."
    )


def compare_top_p(prompt: str) -> None:
    print(f'Prompt: "{prompt}"\n')
    print(
        "Running the SAME prompt at a FIXED temperature=0.9, changing only "
        "top_p — this isolates top_p's effect from temperature's:\n"
    )
    for p in (0.1, 0.5, 1.0):
        print(f"--- top_p={p} (temperature held at 0.9) ---")
        print(ask(prompt, temperature=0.9, top_p=p, top_k=40))
        print()
    print(
        "top_p=0.1: only samples from the smallest set of tokens whose "
        "combined probability reaches 10% — a narrow, high-confidence pool, "
        "even though temperature=0.9 would otherwise allow a lot of variety.\n"
        "top_p=1.0: no narrowing at all — every token the model assigns any "
        "probability to is a candidate, so temperature alone controls the "
        "spread.\n"
        "This is why top_p is often described as narrowing the CANDIDATE "
        "POOL before sampling, while temperature reshapes the PROBABILITIES "
        "within whatever pool is being sampled from — two different knobs on "
        "the same underlying distribution, not two names for the same thing."
    )


def tokenize(prompt: str) -> None:
    """Show the REAL token IDs this model's own tokenizer assigns to
    `prompt` (via Ollama's /api/generate, whose `context` field returns
    the actual integer IDs fed to the model — not an approximation from a
    different tokenizer), then stream generation one token at a time so
    each chunk of output is visibly a token, not a word.

    think=False here too, and for the same reason as ask() above: without
    it, the streamed tokens would (mostly or entirely) be the model's
    invisible reasoning pass, not the visible answer this demo is about.
    """
    # num_predict=1 means "generate almost nothing" — the only thing this
    # call is here for is the `context` list, which is the *input* prompt's
    # token IDs. The model's own template wrapping (system-prompt scaffolding,
    # role markers) is included too — this is genuinely everything the model
    # sees as input, not just the words you typed.
    probe = ollama.generate(model=LOCAL_MODEL, prompt=prompt, think=False, options={"num_predict": 1})
    print(f'Prompt: "{prompt}"\n')
    print(f"Token count for this prompt (as the model's tokenizer sees it): {len(probe.context)}")
    print(f"Raw token IDs: {probe.context}\n")
    print(
        "Notice the count is larger than the word count, and includes IDs "
        "that aren't part of your prompt at all — those are the model's "
        "own chat template (role markers, a system-prompt scaffold Ollama "
        "adds automatically). This is the literal unit your `tokens_in` "
        "cost is billed in — not words, not characters, not even only the "
        "text you wrote.\n"
    )
    print(f"Now generating a real response, one token at a time, streamed:\n")
    print(f'Prompt: "{prompt}"')
    stream = ollama.generate(model=LOCAL_MODEL, prompt=prompt, think=False, options={"num_predict": 15}, stream=True)
    for i, chunk in enumerate(stream):
        if chunk.response:
            print(f"  [token {i}] {chunk.response!r}")
    print(
        "\nEach line above is (usually) exactly one token arriving from the "
        "model — notice some are whole words with a leading space (' the'), "
        "some are word fragments with no space ('over'), and some are just "
        "punctuation or a suffix (\"'s\"). None of that maps cleanly onto "
        "'one word' — a token is whatever chunk the tokenizer's vocabulary "
        "happened to learn, and English words often split unpredictably."
    )


def thinking(prompt: str) -> None:
    """Show this model's separate "thinking" field next to its final
    answer — two genuinely different fields in the same API response, not
    the same text shown twice. think=True here is the deliberate opposite
    of every other subcommand in this file, which explicitly turns
    thinking OFF so it doesn't confound the temperature/top-p/tokenize
    demos.
    """
    print(f'Prompt: "{prompt}"')
    print(f"Model: {LOCAL_MODEL}, think=True\n")
    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=True,
    )
    print("--- thinking (the model's reasoning, before it answers) ---")
    print(response.message.thinking or "(no thinking content returned)")
    print("\n--- content (the actual final answer) ---")
    print(response.message.content or "(no content returned)")
    print(
        "\nThese are two separate fields in the SAME response, not the same "
        "text split in two — `thinking` is the model working through the "
        "problem before committing to an answer; `content` is what actually "
        "gets returned as 'the response' in a normal chat call. A model "
        "without thinking support only ever produces `content` — there's no "
        "hidden reasoning step you're just not seeing, the mechanism genuinely "
        "isn't there. This is also why thinking tokens count against your "
        "token budget even though they're not the visible answer — see "
        "Stage 2's context-budget note in docs/stage-02-llm-fundamentals.md."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_same = sub.add_parser("same-prompt", help="Run one prompt N times at fixed settings.")
    p_same.add_argument("--prompt", default=DEFAULT_PROMPT)
    p_same.add_argument("--temperature", type=float, default=0.0)
    p_same.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    p_same.add_argument("--top-k", type=int, default=40, dest="top_k")
    p_same.add_argument("--runs", type=int, default=3)

    p_temps = sub.add_parser("compare-temps", help="Run one prompt at temperature 0 / 0.5 / 1.0.")
    p_temps.add_argument("--prompt", default=DEFAULT_PROMPT)

    p_topp = sub.add_parser("compare-top-p", help="Run one prompt at top_p 0.1 / 0.5 / 1.0.")
    p_topp.add_argument("--prompt", default=DEFAULT_PROMPT)

    p_tok = sub.add_parser("tokenize", help="Show real token IDs for a prompt, then stream generation token-by-token.")
    p_tok.add_argument("--prompt", default=DEFAULT_PROMPT)

    p_think = sub.add_parser("thinking", help="Show the model's separate thinking vs. content fields.")
    p_think.add_argument("--prompt", default=DEFAULT_MATH_PROMPT)

    args = parser.parse_args()

    print(f"  [model] local via Ollama: {LOCAL_MODEL}\n", file=sys.stderr)

    if args.command == "same-prompt":
        same_prompt(args.prompt, args.temperature, args.top_p, args.top_k, args.runs)
    elif args.command == "compare-temps":
        compare_temps(args.prompt)
    elif args.command == "compare-top-p":
        compare_top_p(args.prompt)
    elif args.command == "tokenize":
        tokenize(args.prompt)
    elif args.command == "thinking":
        thinking(args.prompt)


if __name__ == "__main__":
    main()
