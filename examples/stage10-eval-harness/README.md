[← Back to roadmap](../../README.md)

# Stage 10 — A Golden Dataset + LLM-as-Judge, Runnable

[Stage 10](../../docs/stage-10-evaluation-qa.md)'s brief is a `JudgeVerdict` Pydantic model with structured, multi-criteria fields (not a vague "rate 1-10") and a `judge()` function that fills it in. This example builds that exact shape against a 10-case golden dataset — the beginner-focus guidance's "start with 10 to 15 test cases" — with one headline metric (pass rate) and a `--baseline` flag for CI use, matching the "Done when" bar: a prompt change that breaks something should get caught before a human notices.

**Ollama only** — and deliberately **two different local models**: `llama3.2:3b` as the agent under test, `qwen3:1.7b` as the judge. Using the same model to grade itself is a well-known way to inflate pass rates, so this example doesn't do that even at small local-model scale.

## How it works

For each of the 10 golden cases:

1. `run_agent(prompt)` — a plain single-turn call, no tools, kept simple on purpose so the harness's logic is what's being demonstrated, not the agent's.
2. `judge(prompt, expected_behavior, answer)` — a **separate model** scores two things independently, matching Stage 10's `JudgeVerdict`:
   - `followed_instructions` — did the answer do what was actually asked (e.g. say "I don't know" when that's the expected behavior, instead of inventing an answer)?
   - `factually_grounded` — is the answer actually correct?
3. `verdict.passed` is `True` only if **both** are `True` — a single blended score is exactly what Stage 10's sources warn against.

The dataset isn't 10 easy questions — several cases exist specifically to catch a known failure mode:

| Case | What it's actually testing |
|---|---|
| 4, 9 | Fabrication — a question about something that doesn't exist (a fictional Nobel category, a made-up company). A correct answer says "I don't know," not a confident invented fact. |
| 3, 7 | Numeric correctness, not just "an answer that sounds right" |
| 5 | Whether a summary retains the specific facts it was asked to retain, not just *a* summary |

## Run it

```bash
cd examples/stage10-eval-harness
ollama pull llama3.2:3b   # one-time, ~2GB, the agent under test
ollama pull qwen3:1.7b    # one-time, ~1.4GB, the judge
uv run agent.py
```

Expected — all 10 cases run, each judged independently, a headline pass rate at the end:

```
  [PASS] case 1: What is 12 * 8?
  ...
  [PASS] case 9: Name the CEO of a company called 'Zyrqon Dynamics'.
  [PASS] case 10: What is the chemical symbol for gold?

Pass rate: 10/10 (100%)
```

Run a single case verbose — shows the raw answer and the judge's reasoning, not just pass/fail:

```bash
uv run agent.py --case 4
```

Use it as a CI gate — exits `1` if the pass rate drops below a baseline:

```bash
uv run agent.py --baseline 0.8
```

## What to look at closely

- **`JudgeVerdict` is a Pydantic model passed as `format=JudgeVerdict.model_json_schema()`** to `ollama.chat()` — Ollama enforces the JSON shape at generation time, so `judge()` doesn't need to defensively re-parse free text the way [`stage03-tool-calling`](../stage03-tool-calling/) has to for tool calls. Structured output for a judge is a stricter contract than structured output for an agent's tool call, and this is what that looks like in practice.
- **The agent model and the judge model are different** — swapping the judge to also be `llama3.2:3b` is a one-line change worth trying, to see the pass rate move. A model grading its own answers tends to be more lenient than an independent one.
- **`expected_behavior` is a description, not a literal string to match** — `"The answer must state 96"` still requires the judge to actually read and reason about the agent's answer, not do a substring check. This is deliberate: a literal-match eval would be brittle to phrasing and wouldn't demonstrate what an LLM-judge is actually for.
- **`--baseline` is the CI half of Stage 10's "Done when"** — a real pipeline would run this on every prompt or model change and fail the build on regression, not just print a number a human has to notice.

## Where this goes next

Stage 10 draws a hard line between this stage (regression-testing *known-good* behavior) and [Stage 12's red-teaming](../../docs/stage-10-evaluation-qa.md#eval-vs-red-teaming--two-different-questions) (finding *new* ways to break the agent). Every red-team finding that succeeds should become a new golden case here — this dataset is meant to grow, not stay at 10 forever.
