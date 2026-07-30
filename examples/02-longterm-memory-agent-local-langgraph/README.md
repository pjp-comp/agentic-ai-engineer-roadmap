[← Back to roadmap](../../README.md)

# Example 02c — A Long-Term Memory Agent (Durable Facts, Any Window, Any Session)

Adds the third memory tier from [Stage 4 — Memory + State Management](../../docs/stage-04-memory-state.md) on top of [`examples/02-memory-agent-local-langgraph/`](../02-memory-agent-local-langgraph/): a **durable fact store** that survives regardless of the short-term window size, a `--reset`, or starting a brand-new conversation days later. This is the direct fix for the gap that example leaves on purpose — "the agent forgot something I told it to remember" — by giving the model an explicit `save_fact` tool and always injecting saved facts into the system prompt.

## What it is

- **`ShortTermMemory` + `PersistentState`** — identical to [`examples/02-memory-agent-local-langgraph/`](../02-memory-agent-local-langgraph/): a bounded 20-turn window, checkpointed to `.chat_state.json` after every turn.
- **`LongTermMemory`** (new) — a flat JSON file (`.long_term_facts.json`) of `key: value` facts, completely separate from the turn-by-turn chat log. The model decides what's worth saving and calls `save_fact(key, value)`; every future turn's system prompt includes a "Known facts" block built from this file, so a fact saved in session 1 is visible in session 47, no matter how much has been said in between.
- **Saving with the same key overwrites, not appends** — a tiny version of the "supersession on write" policy from [Stage 4's forgetting section](../../docs/stage-04-memory-state.md#forgetting-and-eviction--long-term-memory-needs-a-cleanup-policy-too): if the model later saves `name: Prag` it replaces `name: Pragnesh` rather than leaving both facts to confuse future recall.
- **No vector store, no embeddings, here** — same "keep it lightweight" choice as every other `02-*` example in this repo. This is exact key/value recall, not semantic similarity search — see [`examples/02-longterm-memory-vectorstore-local-langgraph/`](../02-longterm-memory-vectorstore-local-langgraph/) for the vector-backed version of this exact example, or [Stage 4's `LongTermMemory` class](../../docs/stage-04-memory-state.md#the-three-memory-types) for the concept in the roadmap doc.

## Token cost: bounded here, unbounded there

The model has no memory of its own between calls — every `.invoke()` is stateless, and the entire context (system prompt + conversation so far) is re-sent from scratch on every single turn. That has different cost implications for the two stores in this example:

- **Short-term chat history is bounded.** `ShortTermMemory` is a `deque(maxlen=WINDOW_SIZE)` — once the window fills, adding a new turn evicts the oldest one, so the conversation part of the prompt never exceeds ~20 turns' worth of tokens no matter how long you chat. Cost per call plateaus; it doesn't keep climbing.
- **Long-term facts are NOT bounded.** `as_context_block()` dumps *every* saved fact into the system prompt, on every call, and nothing ever removes a fact except `--forget` wiping all of them. Save 5 facts, every future call pays for 5 facts' worth of tokens. Save 500, every future call pays for 500 — this is a genuine, unaddressed gap in this example, not a hidden one. It's exactly the problem [Stage 4](../../docs/stage-04-memory-state.md) opens with ("naive 'append everything to the prompt' degrades both cost and accuracy") and what its [forgetting/eviction section](../../docs/stage-04-memory-state.md#forgetting-and-eviction--long-term-memory-needs-a-cleanup-policy-too) (TTL, decay, supersession) exists to fix — this example only implements same-key supersession, not the other two.
- **The fix is retrieval, not a bigger window.** A production version wouldn't inject *all* facts into every prompt — it would embed each fact and retrieve only the ones relevant to the current question (the "real semantic recall" item below), so cost scales with *relevant* facts, not total facts ever saved.

## Forgetting policies, explained simply

This example only does one small piece of "forgetting" (overwrite a fact when it's saved again under the same key). [Stage 4](../../docs/stage-04-memory-state.md) describes three real policies for cleaning up long-term memory. Here they are without the jargon — think of `LongTermMemory` as a notebook the agent writes facts into:

- **TTL (time-to-live) = "this note expires."** You write a sticky note that says "user's temp password: XYZ" and set a rule: after 30 days, throw it away automatically, whether or not anyone used it. Good for facts that are only true for a limited time (a temporary project, a password, a promo code) or facts you're legally required to delete after a while (e.g. GDPR "right to be forgotten"). This example has **no TTL at all** — a fact saved once sits in `.long_term_facts.json` forever until you run `--forget`.
- **LRU-style decay = "notes nobody reads fade and eventually get thrown out."** Imagine every time a note gets used, you flip it back to the top of the pile; notes that never get touched slowly sink to the bottom. When the pile gets too big, you throw out the ones at the bottom first. This bounds how large memory gets *without* a hard expiry date — useful facts survive indefinitely, boring ones fade naturally. This example has **no decay** — every fact is equally "fresh" forever, and the pile is never trimmed.
- **Supersession on write = "a new note replaces the old note about the same thing."** If the notebook already has "user's email: old@x.com" and the user gives you a new email, you cross out the old line and write the new one — you don't just add a second line and leave both, because then a future reader wouldn't know which one is true. This example **does** implement this one: [`LongTermMemory.remember()`](agent.py) uses the fact's `key` as a dictionary key, so saving `email` again always overwrites the previous `email` value instead of appending a duplicate.

**Why this matters in practice, even for a toy example:** without TTL or decay, this example's fact file can only grow. That's fine for an afternoon of testing; it's exactly the kind of thing that becomes a real cost and quality problem (see "Token cost" above) the moment an agent runs for weeks and accumulates hundreds of facts nobody ever prunes.

## Run it

Same Ollama/Claude setup as the other `02-*` examples — install Ollama, `ollama pull llama3.2:3b`. `USE_LOCAL_MODEL=true` is the default; set it to `false` in your `.env` if you want the Claude API instead.

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

1. **Real semantic recall** — swap the flat JSON file for a vector store and embed each fact, so recall works by *relevance to the current question* instead of "dump every fact into every prompt." Necessary once you have more than a handful of facts — this example's unconditional injection doesn't scale. Already built: [`examples/02-longterm-memory-vectorstore-local-langgraph/`](../02-longterm-memory-vectorstore-local-langgraph/) does exactly this with a local ChromaDB store.
2. **Forgetting/eviction policies** — [Stage 4's forgetting section](../../docs/stage-04-memory-state.md#forgetting-and-eviction--long-term-memory-needs-a-cleanup-policy-too) (TTL, LRU-style decay, supersession-on-write) applies directly here; this example already does same-key supersession, but has no TTL or decay — a fact saved once stays forever until `--forget` wipes everything.
