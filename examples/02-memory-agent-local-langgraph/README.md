[← Back to roadmap](../../README.md)

# Example 02b — A Memory Agent (Short-Term + Persistent), Running Locally

A multi-turn chat agent that makes the first two memory tiers from [Stage 4 — Memory + State Management](../../docs/stage-04-memory-state.md) concrete and watchable: **short-term memory** (a bounded window of recent turns) and **persistent state** (checkpointed to disk so a killed process resumes). Runs against the same free local Ollama model as [`examples/02-basic-agent-local-langgraph/`](../02-basic-agent-local-langgraph/), or the Claude API — same `USE_LOCAL_MODEL` switch.

## What it is

- **`ShortTermMemory`** — a `deque(maxlen=20)` (10 exchanges). Every new turn beyond that pushes the oldest one out. Nothing is summarized — what's inside the window is recalled exactly; what's outside it is gone from the model's context, full stop.
- **`PersistentState`** — the full conversation checkpointed to a local JSON file (`.chat_state.json`, gitignored) after every turn. Kill the process (Ctrl+C) and run it again: it resumes instead of starting over.
- **No long-term memory here — read this before you rely on it for anything real.** This example only implements the first two of Stage 4's three tiers. That means **any fact will eventually be forgotten** once your conversation runs past `WINDOW_SIZE` turns — even facts you explicitly asked it to remember, even across a restart, because `PersistentState` only rehydrates the *last* `WINDOW_SIZE` turns into the live window (older turns stay in the JSON file, but the model never sees them again). If you ask it to "remember" something and then chat for a while, expect it to eventually forget — that's the tier this example is teaching, not a defect. A separate long-term-memory example (durable facts, independent of window position) is planned; see [Stage 4's `LongTermMemory` class](../../docs/stage-04-memory-state.md#the-three-memory-types) if you want to build that piece yourself in the meantime.

## Run it

Same setup as [`examples/02-basic-agent-local-langgraph/`](../02-basic-agent-local-langgraph/) — install Ollama, `ollama pull llama3.2:3b`, set `USE_LOCAL_MODEL=true` in your repo-root `.env` — or leave it `false`/unset to use the Claude API instead.

```bash
cd examples/02-memory-agent-local-langgraph
uv run agent.py
```

It drops you into an interactive chat. Say something worth remembering, then send enough other messages to push it out of the 20-turn window, then ask about it again:

```
you> My secret code is banana77.
you> (chat about anything else for ~10 more exchanges)
you> What was my secret code?
```

Once the "secret code" turn has scrolled past the last 20 messages, the agent will honestly say it doesn't know — because it genuinely no longer has that turn in context. That's not a bug; it's the actual tradeoff **every** bounded-context system makes, shown directly instead of described abstractly. (This is exactly what happens if you chat with it about your name, then a scheduling plan, then keep going for several more exchanges — the older facts quietly drop out first.)

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

## Token cost stays flat, on purpose

The model has no memory of its own between calls — [`llm.invoke(_to_lc_messages(short_term.get()))`](agent.py) re-sends the *entire* current window as input on every single turn; there's no session state inside Ollama or the Claude API carrying context forward for you. That means token cost per call is directly proportional to how many turns are in `short_term.get()` at that moment — which is exactly why `WINDOW_SIZE` matters here: because it's a bounded `deque`, that number never exceeds 20 turns, so cost per call plateaus once you're past turn 20, instead of climbing forever as the conversation grows. A naive version with no cap (just append every turn, never evict) would resend more tokens on every call for as long as the conversation continues, until you eventually hit the model's context-window limit outright — see [Stage 2's context-budget note](../../docs/stage-02-llm-fundamentals.md) and [Stage 4's opening warning](../../docs/stage-04-memory-state.md) about exactly this failure mode.

## What to look at closely

- **`ShortTermMemory.buffer` is a `deque(maxlen=...)`** — the eviction is Python doing the work, not application logic you wrote; that's the entire implementation of "forgetting" at this tier, and it's also what keeps token cost bounded (see above).
- **`PersistentState` checkpoints the *full*, unbounded history** — deliberately different scope from short-term memory. Persistent state's job is "don't lose the conversation on a crash," not "bound what the model sees" — those are two different problems solved by two different tiers, which is exactly the point Stage 4 makes about not merging memory types into one class.
- **`_build_llm()` is called lazily, inside `run_chat()`, not at module import time** — `--reset` never constructs a model client, so clearing state works even if Ollama isn't running or no API key is set. Worth noticing because the naive version (build the client at the top of the file) silently couples an unrelated code path to your model configuration.
- **`short_term.load(turns[-window:])` on resume** — only the last `WINDOW_SIZE` turns from the saved file get loaded into the live short-term window; older turns exist in the JSON file (persistent tier) but are correctly *not* replayed into the model's active context, matching the behavior it would have had if the process had never restarted.

## Where this goes next

This example stops at short-term + persistent on purpose. [`examples/02-longterm-memory-agent-local-langgraph/`](../02-longterm-memory-agent-local-langgraph/) builds directly on this one and adds the third tier: a durable fact store (e.g. "user's name is Pragnesh," "scheduled to start July 30") saved explicitly via a `save_fact` tool and injected into every turn's context regardless of where it sits in the conversation history — so it survives long past `WINDOW_SIZE` and even into a brand-new conversation days later. That maps to the `LongTermMemory` class in [Stage 4](../../docs/stage-04-memory-state.md#the-three-memory-types); [Stage 4's forgetting section](../../docs/stage-04-memory-state.md#forgetting-and-eviction--long-term-memory-needs-a-cleanup-policy-too) (TTL, LRU-style decay, supersession-on-write) is the natural follow-up on top of that.
