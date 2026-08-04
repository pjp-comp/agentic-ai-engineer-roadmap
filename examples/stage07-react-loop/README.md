[← Back to roadmap](../../README.md)

# Stage 7 — The ReAct Loop, With an Iteration Cap and a Stagnation Check

[Stage 7](../../docs/stage-07-single-agent.md) gives the ReAct loop as pseudocode: think, act, check for stagnation, force a reflection step if stuck, degrade gracefully if the step limit is hit. This example builds that pseudocode as a real, runnable `while` loop against Ollama — no framework, no LangGraph. Stage 7's brief is about the loop's *mechanics*, so this example stays at the same raw-API level as [`stage03-tool-calling`](../stage03-tool-calling/), before any framework abstracts the loop away.

**Ollama only** — `llama3.2:3b`, no Claude path.

## How it works

The agent has one tool, `lookup(city)`, deliberately built to be unreliable: the **first** call for any known city fails ("service busy"), the **second** succeeds. Cities not in the fake dataset never succeed. This isn't a toy detail — it's what forces every piece of Stage 7's brief to actually run instead of the loop finishing cleanly in one pass:

- **`MAX_ITERS`** — asking about a city with no data at all can never succeed. The loop must stop instead of spinning forever, and hand back a fallback answer.
- **Stagnation check** — if the model calls `lookup` with the exact same arguments twice in a row (comparing action *and* result, so a retry that gets a new answer doesn't count as stuck), a forced `reflect()` step runs before the model gets to act again.
- **Graceful degradation** — hitting `MAX_ITERS` returns whatever partial results the loop collected, never a crash and never silence.

```
think() -> is_stagnant()? -> [reflect() -> think() again] -> execute -> record -> loop
```

## A real small-model limitation this surfaced

`llama3.2:3b`, once given a tool, kept calling it again even *after* a successful result — even with an explicit "stop calling tools now" instruction in the prompt. Prompting alone didn't override the model's bias to use an available tool. The fix is at the harness level, not the prompt level: once the loop has a successful (non-error) result in hand, `think()` is called with `offer_tool=False`, which stops passing `tools=` to `ollama.chat()` entirely for that turn — forcing a text-only final answer instead of hoping the model self-regulates. This is the same class of fix as [`stage03-tool-calling`](../stage03-tool-calling/)'s defensive JSON-parsing: a real model quirk, fixed in code that controls the model's options, not in prose the model might ignore.

## Run it

```bash
cd examples/stage07-react-loop
ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
uv run agent.py "What's the weather in Paris?"
```

Expected — one failed attempt, a successful retry, then a clean final answer with no further tool calls:

```
  [step 1] lookup({'city': 'Paris'})
  [step 1] -> error: weather service busy, try again
  [reflect] stuck repeating the same action: called lookup({'city': 'Paris'}) -> error: weather service busy, try again
  [step 2] lookup({'city': 'Paris'})
  [step 2] -> Paris: 14C, light rain
  [step 3] final answer

The current weather in Paris is 14 degrees Celsius and there is light rain.
```

Force the `MAX_ITERS` / graceful-degradation path with a city that will never resolve:

```bash
uv run agent.py "What's the weather in Atlantis?"
```

The loop retries, tries variations, hits the step cap, and returns a partial answer instead of crashing or looping forever.

## What to look at closely

- **`is_stagnant()` compares `(name, args)` from the *last* history entry only** — not the whole history — and doesn't look at the result at all for equality, just the action. A retry of the same city that gets a *different* result (busy → success) is progress, so it's never flagged as stagnant even though the arguments repeat.
- **`offer_tool=False` is the actual fix for the small-model tool-happy bias described above** — this is worth reading before assuming "just tell the model to stop" is enough anywhere else in this repo's examples.
- **The reflection step is a separate, smaller model call**, not a re-run of the main loop — it only sees the last two history entries, on purpose, so it stays cheap and focused on "you're stuck, what now" rather than re-reasoning the whole task.
- **`degrade_gracefully()` never returns an empty string or raises** — even the worst case (no results at all) returns a plain sentence saying so.

## Where this goes next

This is the harness+loop half of the [harness/loop/graph framework](../../docs/stage-07-single-agent.md#harness-loop-and-graph--the-current-framing-for-how-agents-get-built) — one well-looped agent. [`stage08-supervisor-langgraph`](../stage08-supervisor-langgraph/) is the graph half: the same loop concept, but with more than one agent and a router deciding which one runs next.
