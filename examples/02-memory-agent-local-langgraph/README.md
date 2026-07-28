[← Back to roadmap](../../README.md)

# Example 02b — A Memory Agent (Short-Term + Persistent), Running Locally

A multi-turn chat agent that makes the first two memory tiers from [Stage 4 — Memory + State Management](../../docs/stage-04-memory-state.md) concrete and watchable: **short-term memory** (a bounded window of recent turns) and **persistent state** (checkpointed to disk so a killed process resumes). Runs against the same free local Ollama model as [`examples/02-basic-agent-local-langgraph/`](../02-basic-agent-local-langgraph/), or the Claude API — same `USE_LOCAL_MODEL` switch.

## What it is

- **`ShortTermMemory`** — a `deque(maxlen=6)` (3 exchanges). Every new turn pushes the oldest one out. Nothing is summarized — what's inside the window is recalled exactly; what's outside it is gone from the model's context, full stop.
- **`PersistentState`** — the full conversation checkpointed to a local JSON file (`.chat_state.json`, gitignored) after every turn. Kill the process (Ctrl+C) and run it again: it resumes instead of starting over.
- **No long-term (vector) memory here on purpose** — that tier needs a vector store dependency this repo doesn't otherwise pull in. See the `LongTermMemory` class in [Stage 4's doc](../../docs/stage-04-memory-state.md#the-three-memory-types) if you want to extend this example yourself; the short-term/persistent split is the part that's easiest to *feel* by actually running something, so that's what this example focuses on.

## Run it

Same setup as [`examples/02-basic-agent-local-langgraph/`](../02-basic-agent-local-langgraph/) — install Ollama, `ollama pull llama3.2:3b`, set `USE_LOCAL_MODEL=true` in your repo-root `.env` — or leave it `false`/unset to use the Claude API instead.

```bash
cd examples/02-memory-agent-local-langgraph
uv run agent.py
```

It drops you into an interactive chat. Try this to watch the window evict a fact:

```
you> My secret code is banana77.
you> Hi
you> How are you
you> Tell me a fact about the moon
you> What is 5+5?
you> What was my secret code?
```

By the sixth exchange, the first turn ("My secret code is banana77") has been pushed out of the 6-turn window — the agent will honestly say it doesn't know, because it genuinely no longer has that turn in context. That's not a bug being demonstrated; it's the actual tradeoff **every** bounded-context system makes, shown directly instead of described abstractly.

Now try persistence — say something, quit, and restart:

```bash
uv run agent.py
you> My favorite color is teal.
you> quit
uv run agent.py
you> What is my favorite color?
```

The second run prints `[memory] resumed N turn(s) from .chat_state.json` and correctly recalls "teal" — the process restarted, but the conversation didn't.

Clear saved state and start over:

```bash
uv run agent.py --reset
```

## What to look at closely

- **`ShortTermMemory.buffer` is a `deque(maxlen=...)`** — the eviction is Python doing the work, not application logic you wrote; that's the entire implementation of "forgetting" at this tier.
- **`PersistentState` checkpoints the *full*, unbounded history** — deliberately different scope from short-term memory. Persistent state's job is "don't lose the conversation on a crash," not "bound what the model sees" — those are two different problems solved by two different tiers, which is exactly the point Stage 4 makes about not merging memory types into one class.
- **`_build_llm()` is called lazily, inside `run_chat()`, not at module import time** — `--reset` never constructs a model client, so clearing state works even if Ollama isn't running or no API key is set. Worth noticing because the naive version (build the client at the top of the file) silently couples an unrelated code path to your model configuration.
- **`short_term.load(turns[-window:])` on resume** — only the last `WINDOW_SIZE` turns from the saved file get loaded into the live short-term window; older turns exist in the JSON file (persistent tier) but are correctly *not* replayed into the model's active context, matching the behavior it would have had if the process had never restarted.

## Where this goes next

Two natural extensions, both sketched already in [Stage 4](../../docs/stage-04-memory-state.md):

1. **Long-term memory** — swap `PersistentState`'s flat JSON file for a real vector store and add the `LongTermMemory.remember()`/`recall()` methods from the stage doc, so facts survive not just a restart but an entirely new conversation days later.
2. **Forgetting/eviction policies** — once you have long-term memory, [Stage 4's forgetting section](../../docs/stage-04-memory-state.md#forgetting-and-eviction--long-term-memory-needs-a-cleanup-policy-too) covers TTL, LRU-style decay, and supersession-on-write — the same "old turns fall off" idea this example shows at the short-term tier, applied to a store that doesn't naturally bound itself the way a `deque` does.
