[← Back to roadmap](../../README.md)

# Example 02c — A Long-Term Memory Agent (Durable Facts, Any Window, Any Session)

Adds the third memory tier from [Stage 4 — Memory + State Management](../../docs/stage-04-memory-state.md) on top of [`examples/02-memory-agent-local-langgraph/`](../02-memory-agent-local-langgraph/): a **durable fact store** that survives regardless of the short-term window size, a `--reset`, or starting a brand-new conversation days later. This is the direct fix for the gap that example leaves on purpose — "the agent forgot something I told it to remember" — by giving the model an explicit `save_fact` tool and always injecting saved facts into the system prompt.

## What it is

- **`ShortTermMemory` + `PersistentState`** — identical to [`examples/02-memory-agent-local-langgraph/`](../02-memory-agent-local-langgraph/): a bounded 20-turn window, checkpointed to `.chat_state.json` after every turn.
- **`LongTermMemory`** (new) — a flat JSON file (`.long_term_facts.json`) of `key: value` facts, completely separate from the turn-by-turn chat log. The model decides what's worth saving and calls `save_fact(key, value)`; every future turn's system prompt includes a "Known facts" block built from this file, so a fact saved in session 1 is visible in session 47, no matter how much has been said in between.
- **Saving with the same key overwrites, not appends** — a tiny version of the "supersession on write" policy from [Stage 4's forgetting section](../../docs/stage-04-memory-state.md#forgetting-and-eviction--long-term-memory-needs-a-cleanup-policy-too): if the model later saves `name: Prag` it replaces `name: Pragnesh` rather than leaving both facts to confuse future recall.
- **No vector store, no embeddings** — same "keep it lightweight" choice as every other `02-*` example in this repo. This is exact key/value recall, not semantic similarity search. See [Stage 4's `LongTermMemory` class](../../docs/stage-04-memory-state.md#the-three-memory-types) for the vector-backed version if you want to extend this into real semantic recall.

## Run it

Same Ollama/Claude setup as the other `02-*` examples — install Ollama, `ollama pull llama3.2:3b`, set `USE_LOCAL_MODEL=true` in your repo-root `.env` (or leave it `false`/unset for the Claude API).

```bash
cd examples/02-longterm-memory-agent-local-langgraph
uv run agent.py
```

Try the exact scenario the short-term-only example can't handle:

```
you> my name is pragnesh, please remember it
you> we're scheduling something to start on the 30th of july, remember that too
you> quit
```

Then clear only the chat log (not the facts) and start over, simulating a brand-new session:

```bash
uv run agent.py --reset
uv run agent.py
you> hi, do you remember my name and what we scheduled?
```

It does — because `--reset` only clears `.chat_state.json` (the short-term/persistent tier), not `.long_term_facts.json`. That's the whole point: long-term memory is a genuinely separate store from the conversation history, so clearing one doesn't touch the other.

Other flags:

```bash
uv run agent.py --facts     # print all saved long-term facts and exit
uv run agent.py --forget    # clear long-term facts only (chat log untouched)
```

## What to look at closely

- **`save_fact` is a real tool, bound with `.bind_tools(TOOLS)`** — same mechanism as [`examples/01-basic-agent-langgraph/`](../01-basic-agent-langgraph/)'s `calculate` tool. Long-term memory here isn't a framework feature; it's a normal tool call whose side effect happens to be "write to a file the system prompt reads from."
- **Two model instances, not one** — `_build_llm()` returns `(llm, llm_text_only)`: the same underlying model, but only `llm` has `save_fact` bound. After a tool call, the follow-up "now answer the user" turn is sent through `llm_text_only`. This isn't cosmetic — a small local model (Llama 3.2 3B) that still has `tools` bound will often re-issue the *same* tool call again instead of producing a text reply, even when the prompt explicitly tells it to just answer. Removing the tool binding for that second pass makes a text reply the only possible output, instead of relying on prompting alone.
- **The system prompt gives concrete save/don't-save examples, not just a rule** — a one-line instruction ("don't call it for small talk") was not enough for the local model to reliably follow; see "A real limitation" below for what still gets through even with examples.
- **Facts are injected into *every* system prompt, unconditionally** — there's no retrieval step (no "is this fact relevant to the current question?" filter) because there's no similarity search here, just a flat list. Fine for a handful of facts; this is exactly where a real vector store starts to earn its complexity — see "Where this goes next."
- **`--reset` vs. `--forget` are deliberately separate flags** — clearing conversation history and clearing durable facts are different operations with different blast radii, matching Stage 4's point that short-term/persistent/long-term are different tiers with different lifetimes, not one memory system.

## A real limitation, shown honestly

Running the same conversation against both model paths surfaces a genuine, reproducible gap, not a hidden one:

- **Local (Llama 3.2 3B)**: reliably calls `save_fact` for genuine facts (name, a scheduled date), and the two-instance fix above stops it from replacing its actual answer with tool-call narration. It still occasionally re-saves a fact that's already present in "Known facts" instead of recognizing it's redundant — harmless (same key, same value, a no-op overwrite) but a real sign the model isn't checking its own context before acting.
- **Claude (claude-haiku-4-5)**: correctly saves only genuine new facts, skips small talk and one-off answers (arithmetic, "ok thanks") entirely, and never re-saves something already known.

This isn't a bug in this example's code — both models saw the identical tool description and system prompt. It's a real capability gap: smaller open-weight models are less reliable at judging *when* a tool should fire and whether it's already been fired, not just at formatting the call correctly. Worth knowing before you trust a small local model's tool-call judgment in anything beyond a learning exercise — `save_fact` is safe to over-call (worst case, a redundant overwrite), but a tool with real side effects (send an email, charge a card) being re-triggered by an eager small model is a genuine production risk, not a curiosity.

If you hit an agent that answers *completely* off-topic after a tool call (e.g. replying to "ok thanks" with "I've saved your fact" instead of acknowledging the "ok thanks"), that's the bug the two-instance fix above addresses — if you still see it, the model likely needs the same tools-unbound-follow-up treatment applied somewhere it isn't yet.

## Where this goes next

Two directions from here, both flagged already in Stage 4:

1. **Real semantic recall** — swap the flat JSON file for a vector store (pgvector, Chroma) and embed each fact, so recall works by *relevance to the current question* instead of "dump every fact into every prompt." Necessary once you have more than a handful of facts — this example's unconditional injection doesn't scale.
2. **Forgetting/eviction policies** — [Stage 4's forgetting section](../../docs/stage-04-memory-state.md#forgetting-and-eviction--long-term-memory-needs-a-cleanup-policy-too) (TTL, LRU-style decay, supersession-on-write) applies directly here; this example already does same-key supersession, but has no TTL or decay — a fact saved once stays forever until `--forget` wipes everything.
