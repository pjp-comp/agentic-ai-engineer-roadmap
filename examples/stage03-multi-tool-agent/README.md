[← Back to roadmap](../../README.md)

# Example 01b — A Multi-Tool AI Agent

Same ReAct loop as [`examples/stage03-tool-calling/`](../stage03-tool-calling/), but with four tools instead of one. The basic example proves the model *can* call a tool; this one proves it can choose the *right* one — the part that actually matters once an agent has more than a single job.

**Runs against a free local Ollama model by default** (`USE_LOCAL_MODEL=true`) — same switch as every other example, set `USE_LOCAL_MODEL=false` for Claude instead. Worth knowing: testing this example locally surfaced two real local-model quirks. First, Llama 3.2 3B sometimes sends `convert_units`' numeric `value` as a string (`'5'`) instead of a number, where Claude follows the JSON schema strictly — `convert_units()` now coerces defensively (`float(value)`) rather than trusting either model's schema compliance. Second, on a multi-step task it can write a *correct* next tool call as plain text instead of actually issuing it — `_extract_written_out_tool_call()` in `agent.py` recognizes that specific shape (a known tool name, all required arguments present) and runs it directly rather than asking the model to retry (see `examples/stage03-tool-calling/README.md` for the full writeup, including what this does and doesn't fix).

## What it is

- **Four tools**: `calculate` (arithmetic), `word_count` (text), `convert_units` (length conversion between m/km/mi/ft), `get_weather` (a fake, fixed-data lookup that can return a not-found error on purpose).
- **One loop**: identical shape to example 01 — send messages → if the model asks for a tool (or several), run them and send results back → repeat, capped at `MAX_ITERATIONS`.
- **A dispatch table** (`_DISPATCH`) instead of an `if/elif` chain — this is the one structural difference from example 01, and it's the natural shape once you have more than two or three tools.

This is still an **AI agent**, not agentic AI — see the [distinction in the main README](../../README.md#ai-agent-vs-agentic-ai--the-distinction-that-matters). More tools doesn't mean more autonomy; it's still one model, one bounded task, one loop.

## Run it

Managed with [`uv`](https://docs.astral.sh/uv/), same as example 01.

```bash
cp ../../.env.example ../../.env   # skip if you already did this for example 01 — same key, shared at repo root
```

```bash
cd examples/stage03-multi-tool-agent
uv run agent.py "How many words are in 'the quick brown fox jumps'? Also convert 5 km to miles. Also what's the weather in Tokyo?"
```

Expected output (tool calls print to stderr so stdout stays clean):

```
  [tool call] word_count({'text': 'the quick brown fox jumps'})
  [tool call] convert_units({'value': 5, 'from_unit': 'km', 'to_unit': 'mi'})
  [tool call] get_weather({'city': 'Tokyo'})
Here are the results:

1. **Word count**: "the quick brown fox jumps" contains **5 words**
2. **Unit conversion**: 5 km = **3.11 miles** (approximately)
3. **Weather in Tokyo**: **27°C, sunny**
```

All three tools get called in a single turn here — that's the multi-`tool_use`-block batching from example 01, now visibly doing real work with more than one tool in the mix.

Try the error path on purpose:

```bash
uv run agent.py "What's the weather in Atlantis?"
```

`get_weather` returns `Error: no weather data for 'Atlantis' — known cities: Paris, Tokyo` as a `tool_result`, and the model recovers with a plain-text explanation instead of crashing — the same `is_error`-shaped recovery pattern from [Stage 3](../../docs/stage-03-tool-calling.md), just triggered by a real "not found" instead of a validation error.

<details>
<summary>Without <code>uv</code> (plain <code>pip</code>)</summary>

```bash
cd examples/stage03-multi-tool-agent
python -m venv .venv && source .venv/bin/activate
pip install anthropic python-dotenv
export ANTHROPIC_API_KEY=sk-ant-...
python agent.py "How many words are in 'the quick brown fox jumps'? Also convert 5 km to miles."
```
</details>

## What to look at closely

- **`_DISPATCH`, not a growing `if/elif`** — each tool name maps to a one-line lambda calling the real function. Adding tool #5 means adding one dict entry and one `TOOLS` schema, not touching the loop.
- **The system prompt now says "pick whichever tool actually fits"** — with one tool, the model has no real choice to make. With four, prompting for restraint matters: nothing stops the model from calling `calculate` on a word-count question if the tool descriptions are vague. Precise `description` fields in `TOOLS` do more work here than the system prompt does.
- **`get_weather`'s deliberate error path** — a real tool set always includes at least one tool that can fail for reasons that aren't the caller's fault (not-found, rate-limited, offline). Watch it recover in the transcript rather than assuming it will.
- **Still one `tool_results` batch per turn** — the loop logic is byte-for-byte the same as example 01's; more tools didn't require more loop complexity, only more dispatch and better descriptions.
- **`tool_name` on every tool result** — with one tool you can omit it and nothing breaks. With three results coming back from a single turn, omitting it hands the model an anonymous list and it has to *guess* which result answers which call. When it guesses wrong the symptom is bizarre rather than obviously broken: the agent starts answering a question nobody asked. This is the first example in the repo where that field stops being optional.

## A real limit of small models, worth seeing

Run this example four or five times. Roughly half the runs answer all three questions cleanly; the rest drop one, or substitute a canned "I don't have real-time weather access" refusal *for a result the tool already returned successfully*. **This is not a bug in the code, and it's deliberately not hidden.**

Two things were verified before concluding that, and both are worth knowing as debugging technique:

1. **The mechanism is provably correct.** Drive the loop by hand and all three tools dispatch, all three results come back correctly labelled (`5`, `3.107`, `27°C, sunny`), and the model composes the right answer. The plumbing works.
2. **The obvious harness fix doesn't help here.** [Stage 7](../../docs/stage-07-single-agent.md) teaches that once the loop has what it needs, you stop passing `tools=` so the model can't keep calling them. Measured over five paired trials, that changes nothing on this task — and occasionally makes it worse (one run confidently reported "9 words"). A real fix has to survive being measured; this one didn't, so it isn't applied.

What's left is the model's actual ceiling. Three simultaneous calls plus three results is near the limit of what a 3B model coordinates reliably — the same loop is stable with Claude (`USE_LOCAL_MODEL=false`) and stable locally with *one* tool (example 01). Neither the loop nor the model is at fault alone; it's their interaction with a harder coordination task.

The lesson that generalizes: **tool-calling reliability degrades with the number of concurrent calls, not only with model size.** If you need many tools on a small model, the fix is architectural rather than prompt-level — show a relevant subset of tools per turn (Stage 2's [context confusion](../../docs/stage-02-llm-fundamentals.md#the-four-context-failure-modes--a-debugging-vocabulary)), or constrain the model to one call per turn and let the loop sequence them. Prompting the model to "be careful" does not close this gap, and neither does the Stage 7 trick.

## Extend it (optional exercises)

1. Add a fifth tool that overlaps in purpose with an existing one (e.g. a second, slightly different `calculate`) and see whether vague descriptions cause the model to pick the wrong one.
2. Force two tools to both plausibly apply to the same question and see which one the model reaches for, and why (read its text reasoning if you ask it to explain).
3. Swap `get_weather`'s fake dict for a real API call and see how the `is_error` recovery pattern holds up against a genuine network failure, not just a scripted one.

## Where this goes next

Same as example 01: this is still a single agent. The next step toward **agentic AI** (Stage 8) isn't more tools on one agent — it's splitting responsibility across *multiple* agents that each own a narrower toolset, coordinated via Stage 6's shared-session pattern. A four-tool agent is often the point where that split starts to make sense: if `get_weather` and `convert_units` belong to a conceptually different "research" role than `calculate`, that's a signal to consider two agents instead of one with a longer tool list.
