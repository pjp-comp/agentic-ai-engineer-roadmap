# Becoming an Agentic AI Engineer

A sequenced, 14-stage curriculum from async Python fundamentals to shipping observable, guarded, production multi-agent systems. Each stage lists what to build, a runnable code brief, a completion check, and vetted sources — official docs first, courses and papers second.

- **Prepared:** 2026-07-17
- **Pace:** ~4–7 months at 8–10 hrs/week

## How to use this

Stages are ordered by dependency, not difficulty — Stage 8 assumes Stages 3, 6, and 7 are solid, since orchestration is just single-agent loops sharing sessions. Don't skip the "brief" in each stage: it's the smallest program that forces you to touch the real API surface, not a toy print statement.

Treat official docs as ground truth and everything else (courses, blog tutorials) as commentary on it — frameworks like LangGraph ship breaking changes often enough that an older tutorial can mislead you on current APIs.

## Beginner Track (Simple)

If you are a beginner, follow this path first and keep the goal narrow: build one stable single-agent application before adding advanced architecture.

The track is split into three phases, because the useful milestone isn't "finished stage N" — it's **"I have a working agent," then "I can trust it," then "I can run it for other people."** Each phase produces something you can demo.

**Phase 1 — get one agent working** (the goal: a ReAct loop you understand line by line)

| # | Stage | Why it's here |
|---|---|---|
| 1 | [Stage 01 — Python + Async Foundations](docs/stage-01-python-async.md) | Every later stage is I/O juggling; skipping this makes everything after it flaky in confusing ways |
| 2 | [Stage 02 — LLM Fundamentals](docs/stage-02-llm-fundamentals.md) | Tokens, sampling, and context engineering — the vocabulary every other stage assumes |
| 3 | [Stage 03 — Tool Calling](docs/stage-03-tool-calling.md) | The single most load-bearing skill in the roadmap. Do not rush it |
| 4 | [Stage 07 — Single-Agent Workflows](docs/stage-07-single-agent.md) | The ReAct loop itself. **You have a working agent at the end of this stage** |

**Phase 2 — make it remember and behave** (the goal: an agent that survives a restart and doesn't do anything alarming)

| # | Stage | Why it's here |
|---|---|---|
| 5 | [Stage 04 — Memory + State](docs/stage-04-memory-state.md) | Now that a loop exists, give it memory. Build short-term first; long-term is optional on the first pass |
| 6 | [Stage 06 — Sessions, State + Events](docs/stage-06-sessions-state.md) | Persistence across restarts — the thing that makes it feel like software rather than a script |
| 7 | [Stage 12 — Security + Guardrails](docs/stage-12-security-guardrails.md) | The moment your agent has tools with real side effects, this stops being optional |

**Phase 3 — prove it works and ship it** (the goal: evidence, not vibes)

| # | Stage | Why it's here |
|---|---|---|
| 8 | [Stage 11 — Observability + Tracing](docs/stage-11-observability.md) | Traces first — you can't evaluate steps you can't see, and Stage 10's trajectory eval reads exactly these records |
| 9 | [Stage 10 — Evaluation + QA](docs/stage-10-evaluation-qa.md) | Now turn those traces into pass/fail assertions |
| 10 | [Stage 13 — Production Deployment](docs/stage-13-deployment.md) | Idempotent tools and a CI gate. Read the agent-native-infra section as a map, not a to-do list |

> **One deviation from the main roadmap's numbering, on purpose:** the beginner track does Stage 07 *before* Stages 04–06, and Stage 11 *before* Stage 10. The main sequence is ordered by dependency; this one is ordered by motivation — memory is much easier to understand once you have a loop that visibly forgets things, and eval is easier once you have traces to assert against. Both orders are correct; use this one if you're learning alone and need a working demo early to stay motivated.
>
> **Skip Stage 11's runnable example on the first pass** — it instruments the Stage 8 supervisor graph, which isn't in this track. Read the doc, apply the tracing idea to your own single agent, and come back to the example after Stage 8.

Optional later (after your first stable project):

- [Stage 05 — RAG + Retrieval](docs/stage-05-rag-retrieval.md) — add this when your agent needs to answer from documents it didn't write
- [Stage 08 — Multi-Agent Orchestration](docs/stage-08-multi-agent.md) — add this only when one well-built agent is demonstrably the bottleneck
- [Stage 09 — Human-in-the-Loop Systems](docs/stage-09-human-in-the-loop.md) — add this the first time an agent action is expensive to undo
- [Stage 14 — Open Source + Portfolio](docs/stage-14-portfolio.md) — the capstone; do it once you have something worth showing

Beginner rules:

- Start with one model.
- Start with one tool.
- Start with one agent.
- Add complexity only after a stable baseline.
- **Finish something before improving anything.** A working ugly agent teaches more than a beautifully-architected half-built one.

## Agent basics — what, why, and how

**What is an agent?** An AI agent is a program where an LLM decides what to do next, instead of a developer hard-coding every step. A normal program follows a fixed path you wrote in advance (`if X then Y`). An agent is given a goal and a set of tools, and *the model itself* chooses which tool to call, in what order, based on what it sees — including deciding it's done and returning an answer.

**Why use one instead of a normal script?** Reach for an agent when the steps needed to solve a task can't be known in advance — the right sequence of actions depends on what earlier steps returned. If you can write the exact steps ahead of time, write a normal script instead; it'll be cheaper, faster, and easier to debug. Agents earn their cost specifically on tasks with unpredictable branching (a question might need 0, 1, or 5 tool calls — you don't know until you're in it).

**How does one actually work?** Every agent, no matter the framework, runs the same loop:

1. **Perceive** — the model reads the current conversation/state (the user's task, plus any tool results so far).
2. **Decide** — the model reasons about what to do next: answer now, or call a tool.
3. **Act** — if it chose a tool, your code runs that tool (a calculator, a web search, a database query...) and captures the result.
4. **Observe & repeat** — the tool's result goes back to the model, and the loop returns to step 1 — until the model has enough information to give a final answer, or a safety limit (max iterations) is hit so it can't loop forever.

This is called the **ReAct pattern** (Reason + Act), and it's the same shape whether you write it by hand or a framework writes it for you.

**Minimal example — the actual decision point**, from [`examples/stage03-tool-calling/agent.py`](examples/stage03-tool-calling/agent.py). This example is built on Claude's API, but the decision point itself is universal: every tool-calling API has some field that tells you "the model wants to call a tool" instead of answering directly — Claude spells it `stop_reason == "tool_use"`, OpenAI spells it `finish_reason == "tool_calls"`, Gemini returns a `functionCall` part instead of text. Same branch, different field name:

**Claude API example:**

```python
response = client.messages.create(model=MODEL, tools=TOOLS, messages=messages)

if response.stop_reason != "tool_use":
    return final_answer          # step 2 decided: no tool needed, done
else:
    result = run_the_tool(...)   # step 3: act
    messages.append(tool_result)  # step 4: observe, loop back to step 1
```

Nothing here is magic — it's a `while` loop with an `if` branch. What frameworks like LangGraph add later is a cleaner way to express that same loop once you have more than one tool or more than one agent (see the three worked examples below, which build the identical agent three different ways so you can see exactly what each layer adds).

## AI Agent vs. Agentic AI — the distinction that matters

These terms get used interchangeably, but they describe different points on a spectrum of autonomy. Knowing which one you're building changes your architecture, your error-handling budget, and how much you can trust the system unattended.

| | **AI Agent** | **Agentic AI** |
|---|---|---|
| **Scope** | One model, one task, a bounded set of tools | Multiple agents/roles, coordinated toward a broader goal |
| **Loop** | Perceive → decide → act, usually a single pass or a short bounded loop | Continuous plan → act → reflect → replan, often spanning many sub-tasks |
| **Autonomy** | Executes a defined task; a human or a fixed pipeline decides what task to run next | Decomposes the *goal itself* — decides what tasks need to exist |
| **State** | Mostly stateless per call, maybe a short conversation buffer | Persistent memory, shared state across agents and sessions |
| **Example** | A support bot that looks up an order and answers one question | A system that plans a trip, books flights, adjusts for a cancellation, and re-plans the itinerary without being told each step |
| **Where it lives in this roadmap** | Stages 1–6 build the parts; **Stage 7** is where a single agent becomes real | **Stage 8** is where agentic AI starts — orchestration, handoffs, and shared goals across agents |

The short version: **an AI agent is a component; agentic AI is a system of components pursuing a goal with minimal supervision.** A single ReAct loop that calls a weather API is an agent. A crew of agents that researches a market, drafts a strategy, and revises it based on a critic agent's feedback is agentic AI. Stage 7 in this roadmap teaches you to build the former well; Stage 8 teaches you to compose several of them into the latter.

## Start here, before any agent: sampling parameters

[`examples/stage02-sampling-params/`](examples/stage02-sampling-params/) isn't an agent — no tools, no loop, just raw calls to a local model, so [Stage 2's mechanics](docs/stage-02-llm-fundamentals.md#how-llms-actually-work) become things you watch happen instead of only read about: temperature/top-p side by side, a prompt's real token IDs plus token-by-token streamed generation, and a model's separate "thinking" field next to its final answer. Every `agent.py` in this repo sets `temperature=0` (or `0.3`) without explaining why on the page — this is where that "why" comes from, before any tool-calling complexity gets added on top.

## Start here: a basic AI agent

Before touching a framework, build one agent from raw API calls so the ReAct loop isn't a mystery a library is hiding from you. [`examples/stage03-tool-calling/`](examples/stage03-tool-calling/) is a ~120-line Python script: one tool (calculator), one loop, no framework. Run it, then read Stage 7 and Stage 8 knowing exactly what "framework magic" is standing in for.

Once that's comfortable, four follow-ups build on it in different directions:

- [`examples/stage03-multi-tool-agent/`](examples/stage03-multi-tool-agent/) — the same loop, unchanged, but with four tools instead of one — shows tool *selection*, not just tool use, plus a dispatch table instead of an `if/elif` chain and a deliberate tool-error recovery path.
- [`examples/stage03-tool-calling-langchain/`](examples/stage03-tool-calling-langchain/) — the same agent using LangChain's `ChatAnthropic` + `@tool`, but still a hand-written loop — makes concrete why LangChain alone doesn't replace the loop.
- [`examples/stage03-tool-calling-langgraph/`](examples/stage03-tool-calling-langgraph/) — the same agent, rebuilt on LangGraph's `StateGraph` instead of the raw loop, with a line-by-line comparison of what the framework replaces.
- [`examples/stage03-mcp-tool-server/`](examples/stage03-mcp-tool-server/) — the same agent again, with its one tool moved behind an MCP server, showing what MCP actually changes (and doesn't) versus the plain tool array in example 01.

Read the first three in order and you get the full picture: raw API → LangChain's components (still your own loop) → LangGraph's graph (the loop becomes edges). The MCP one is orthogonal — it changes where the tool *lives*, not how the loop works.

Once the calculator toy feels solid, two more examples do real agentic work — same task (give it a company name, get a dated, grouped news summary), two different search tools, chosen to make one distinction concrete:

- [`examples/stage03-web-agent-langgraph/`](examples/stage03-web-agent-langgraph/) — search via Claude's built-in `web_search`, a **server-side** tool (the same category as OpenAI's hosted `web_search` tool or Gemini's grounding-with-search): the provider's own infrastructure runs the search, so the graph is just one node — no `ToolNode`, because there's nothing left for your code to execute.
- [`examples/stage03-web-agent-tavily/`](examples/stage03-web-agent-tavily/) — the identical agent and prompt, using [Tavily](https://tavily.com/) instead — a **client-side** tool, so `ToolNode` and the agent↔tools loop are back, same shape as the calculator LangGraph example.

Both refuse to fabricate a summary when search comes up empty — read them back to back and the lesson is: whether a graph needs a `ToolNode` depends entirely on whether the tool runs on your infrastructure or the provider's, not on what the tool does.

The natural next step after that toward *agentic* AI is a two-agent handoff — see the note at the bottom of example 01's README for where that lives once you build it.

## Running the examples

Every example is a self-contained [uv](https://docs.astral.sh/uv/) project — its own `pyproject.toml` and pinned `uv.lock`, no shared virtualenv to corrupt. `cd` into one and run it; `uv` installs what it needs on first run.

```bash
ollama pull llama3.2:3b                  # once, ~2GB — shared by every example
cd examples/stage03-tool-calling && uv run agent.py
```

**The entry point is `agent.py`** in every example but two, both for good reason:

| Example | Entry point | Why |
|---|---|---|
| `stage01-async-fanout` | `uv run uvicorn app:app` | It's a FastAPI service, not a script — `app.py` is that framework's convention, and the point of the stage is a running server you `curl` |
| `stage02-sampling-params` | `uv run sampling_params.py <subcommand>` | Not an agent at all — five separate demos (`compare-temps`, `tokenize`, `thinking`, …). Run it with no arguments and it lists them |

A handful take arguments or subcommands (`--approve`, `--history`, `--trace`, `--rebuild`); each example's own README is authoritative, and running with no arguments does something sensible everywhere.

**A few examples are interactive** (`stage04-memory-agent`, `stage04-longterm-memory-*`, `stage03-mcp-tool-server`, `stage05-multimodal-rag-agent`) — they read from stdin and wait for you. That's deliberate: memory and session behavior only become visible across several turns.

**Three need something beyond Ollama:** `stage03-web-agent-tavily` (a free Tavily API key), `stage03-web-agent-langgraph` (an Anthropic key — it demonstrates Claude's server-side `web_search`), and `stage05-multimodal-rag-agent` (your own PDFs in its `assets/` folder, plus `ollama pull qwen2.5vl:7b`). Everything else runs offline and free.

## Running for free, offline, with an open-weight model

The ReAct loop and tool-calling contract aren't Claude-specific — any tool-calling model can fill that role. **Every runnable example in this repo now defaults to a free local model** (Llama 3.2 3B via [Ollama](https://ollama.com/)), set via `USE_LOCAL_MODEL=true` in `.env.example` — no API key needed to try any of them. Set `USE_LOCAL_MODEL=false` in your `.env` for the Claude API instead; the Claude code path is still fully wired up in every example, just not the default anymore.

Setup, once: `ollama pull llama3.2:3b` (~2GB, one-time). Every example's `_build_llm()` (or its raw-SDK equivalent) branches on the same flag, so switching between local and Claude never requires touching code — just `.env`.

- **Stage 3's tool-calling examples** (`stage03-tool-calling`, `stage03-tool-calling-langchain`, `stage03-tool-calling-langgraph`, `stage03-multi-tool-agent`, `stage03-web-agent-tavily`, `stage03-mcp-tool-server`) — all run local-first now, covering raw-SDK, LangChain, LangGraph, multi-tool dispatch, and MCP discovery. The one exception: [`stage03-web-agent-langgraph`](examples/stage03-web-agent-langgraph/) stays Claude-only, on purpose — its whole point is Claude's *server-side* `web_search` tool, a hosted capability with no local equivalent to fall back to.
- [`examples/stage02-basic-agent-local/`](examples/stage02-basic-agent-local/) — the calculator agent from example 01, rebuilt to name the local/Claude split explicitly — same graph, one `if` branch decides which model answers.
- [`examples/stage04-memory-agent/`](examples/stage04-memory-agent/) — a multi-turn chat agent demonstrating [Stage 4](docs/stage-04-memory-state.md)'s short-term (bounded window) and persistent (checkpoint-to-disk) memory tiers — watch a fact age out of context once the conversation runs past the window, then watch a restart resume correctly.
- [`examples/stage04-longterm-memory-agent/`](examples/stage04-longterm-memory-agent/) — adds Stage 4's third tier on top of the memory example above: a `save_fact` tool and durable JSON fact store, so a fact survives regardless of window size or session boundary — plus an honest, measured comparison of how much more reliably Claude judges *when* to call that tool versus the small local model.
- [`examples/stage04-longterm-memory-vectorstore/`](examples/stage04-longterm-memory-vectorstore/) — the same long-term memory idea, but backed by a real local vector store (ChromaDB) instead of a flat JSON file, so facts are retrieved by *meaning* ("when are we kicking things off?" correctly finds a fact saved as "start ai learning on 30th july," despite sharing no words) — connects directly to [Stage 5's RAG patterns](docs/stage-05-rag-retrieval.md).
- [`examples/stage05-rag-langgraph/`](examples/stage05-rag-langgraph/) — Corrective RAG (retrieve → grade → generate, with a conditional retry-or-refuse edge) as a real LangGraph graph instead of nested function calls — **Ollama only**, both chat and embeddings local, against a synthetic corpus so a correct answer can only come from retrieval.
- [`examples/stage05-multimodal-rag-agent/`](examples/stage05-multimodal-rag-agent/) — the same corrective-RAG loop over real PDFs (bring your own — drop them in `assets/`) with text, tables, and embedded charts, using **multi-vector retrieval**: a generated summary of each table/image gets embedded for search, but the original raw content is what the final answer is generated from. Split into one file per pipeline stage (chunking, vision, embeddings, vector store, retrieval, generation). One shared, growable index across multiple PDFs; integrates Stage 6's persistent sessions, so a research conversation survives a restart. **Ollama only**, including a local vision model (`qwen2.5vl:7b`) for chart reading — a real vision-model swap (from `llava:7b`) is documented in the example's README, and a working, opt-in CLIP path is included for image-similarity search (documented as a deliberately-not-default choice, since CLIP can't read a chart's values the way a vision-language model can).
- [`examples/news-event-storyline-rag/`](examples/news-event-storyline-rag/) — **a design document, and deliberately only that** — the architecture for tracking an evolving news storyline across many articles over time, where plain chunk-retrieval structurally cannot produce the right answer. It's here as a worked example of *architecture written down before code*, which is a skill this roadmap otherwise only asserts is important. Its two lessons (separate article/event/storyline; make entity identity a hard filter, never a similarity contributor) generalize well past news. Connects to [Stage 5](docs/stage-05-rag-retrieval.md#when-rags-unit-of-retrieval-is-wrong--a-worked-design-case).
- [`examples/stage06-sessions-plain/`](examples/stage06-sessions-plain/) — [Stage 6](docs/stage-06-sessions-state.md)'s session/state/events triad with **no framework at all** — the doc's own `Event`/`Session`/`SessionService` classes, made persistent (one JSON file per session). Read alongside the LangGraph version below to see exactly what a framework changes vs. what stays the same either way. **Ollama only.**
- [`examples/stage06-sessions-langgraph/`](examples/stage06-sessions-langgraph/) — the same session/state/events triad built on LangGraph's real persistence primitives instead (`thread_id`, a `SqliteSaver` checkpointer that survives across separate process runs) — a researcher/writer pair that only ever communicate through shared state. **Ollama only.**
- [`examples/stage07-react-loop/`](examples/stage07-react-loop/) — [Stage 7](docs/stage-07-single-agent.md)'s ReAct loop brief (max-iteration cap, Reflexion-style stagnation check, graceful degradation) as a raw `while` loop, no framework — plus a real small-model quirk this surfaced (see the gap note below).
- [`examples/stage08-supervisor-langgraph/`](examples/stage08-supervisor-langgraph/) — [Stage 8](docs/stage-08-multi-agent.md)'s supervisor pattern as a real LangGraph graph: a router node (no LLM call), a researcher/writer worker pair, and a critic node guarding the handoff between them. **Ollama only.**
- [`examples/stage09-human-in-the-loop/`](examples/stage09-human-in-the-loop/) — [Stage 9](docs/stage-09-human-in-the-loop.md)'s approval-gate brief on real LangGraph `interrupt_before` + `SqliteSaver` — a graph pauses before a high-risk action, a human approves or rejects it from a separate process, the graph resumes exactly where it paused. **Ollama only.**
- [`examples/stage10-eval-harness/`](examples/stage10-eval-harness/) — [Stage 10](docs/stage-10-evaluation-qa.md)'s golden-dataset + LLM-as-judge brief: 10 cases, a structured `JudgeVerdict`, a headline pass-rate metric, and a `--baseline` flag for CI use — judged by a *different* local model than the one under test. **Ollama only.**
- [`examples/stage11-observability/`](examples/stage11-observability/) — [Stage 11](docs/stage-11-observability.md)'s tracing brief applied to the Stage 8 supervisor graph: every node emits a real span (input/output, real token counts, latency, cost) to a local trace log, plus a dashboard script for p95 latency/$-per-run. No hosted backend required. **Ollama only.**
- [`examples/stage12-guardrails/`](examples/stage12-guardrails/) — [Stage 12](docs/stage-12-security-guardrails.md)'s guardrail brief: prompt-injection detection + neutralization in tool output, a per-role tool allowlist enforced in code, and a genuinely sandboxed subprocess for code execution — run against a real "ignore previous instructions" attack. **Ollama only.**
- [`examples/stage13-idempotent-tools/`](examples/stage13-idempotent-tools/) — [Stage 13](docs/stage-13-deployment.md)'s idempotency-key and tool-caching brief, with a simulated crash-then-retry that proves a `charge_card` call doesn't double-charge. No LLM.
- [`examples/stage01-async-fanout/`](examples/stage01-async-fanout/) — [Stage 1](docs/stage-01-python-async.md)'s async fan-out brief: three mock tools with different failure modes, `asyncio.gather` + per-call timeout, and a circuit breaker. No LLM — this stage is entirely async plumbing.

**A real gap this surfaced, worth knowing before you lean on a local model:** local models don't always respect tool schemas as strictly as Claude does — testing `stage03-multi-tool-agent` locally, Llama 3.2 3B sent a numeric argument as a string where Claude sent a proper number, breaking naive code that trusted the schema. Tool functions that accept numbers should coerce defensively (`float(value)`, with a clear error on failure) rather than assume any model — local or not — will always honor the declared type.

**A second, from `stage03-multi-tool-agent`:** when a single turn issues several tool calls, each result must be labelled with which call it answers (`tool_name` in Ollama's API, `tool_use_id` in Claude's). With one tool you can omit it and never notice; with three, the model gets an anonymous list, mis-attributes the results, and starts answering a question nobody asked — a failure that looks like model confusion but is a missing field. Even with the field correct, about half of local runs still drop one of the three answers: three concurrent calls sit at the reliability ceiling of a 3B model. Notably, the Stage 7 fix below (stop passing `tools=` once the loop has what it needs) was measured on this task and **did not help** — a reminder that a technique that works in one situation still has to be verified in the next. Tool-calling reliability degrades with the *number of concurrent calls*, not only with model size.

**A third one, from `stage07-react-loop`:** Llama 3.2 3B kept calling a tool again even *after* it already had a successful result — even with an explicit "stop calling tools now" instruction in the prompt. Prompting alone didn't override the model's bias to use an available tool. The fix lives at the harness level, not the prompt level: stop passing `tools=` to the model at all once the loop already has what it needs, forcing a text-only answer instead of hoping the model self-regulates.

## Stages

| # | Stage | Core topics |
|---|-------|--------------|
| 01 | [Python + Async Foundations](docs/stage-01-python-async.md) | asyncio, FastAPI, event-driven architecture, error handling, API integration |
| 02 | [LLM Fundamentals for Agents](docs/stage-02-llm-fundamentals.md) | Context engineering + its four failure modes, model routing, token economics, a cost reference table, latency tradeoffs |
| 03 | [Tool Calling + Structured Outputs](docs/stage-03-tool-calling.md) | Pydantic validation, function schemas, error recovery, MCP vs. plain APIs, Skills + progressive disclosure, computer-use tools |
| 04 | [Memory + State Management](docs/stage-04-memory-state.md) | Short-term, persistent, and long-term memory as three distinct composable pieces |
| 05 | [RAG + Retrieval](docs/stage-05-rag-retrieval.md) | Vector databases, chunking, naive vs. hybrid vs. agentic RAG, corrective RAG (CRAG), retrieval vs. tool calls, when the retrieval *unit* is wrong |
| 06 | [Sessions, State + Events](docs/stage-06-sessions-state.md) | The session object, state vs. events, an in-memory session service, updating context, inter-agent communication via shared sessions |
| 07 | [Single-Agent Workflows](docs/stage-07-single-agent.md) | The six Anthropic workflow patterns, ReAct vs. Plan-and-Execute vs. ReWOO, Tree-of-Thought, self-reflection, iteration limits, graceful degradation, long-running agents, cognitive architecture, computer-use/voice agents |
| 08 | [Multi-Agent Orchestration](docs/stage-08-multi-agent.md) | Six named topologies (supervisor/pipeline/swarm/hierarchical/blackboard/network), sub-agents as context isolation, framework landscape, message passing, handoffs + inline guardrails, A2A protocol |
| 09 | [Human-in-the-Loop Systems](docs/stage-09-human-in-the-loop.md) | The four intervention patterns, deciding what needs a human (reversibility over confidence), approval gates, audit trails, resume logic, approval fatigue |
| 10 | [Evaluation + Quality Assurance](docs/stage-10-evaluation-qa.md) | Automated eval harnesses, LLM-as-judge (and validating the judge), trajectory evaluation, regression testing, eval vs. red-teaming |
| 11 | [Observability + Tracing](docs/stage-11-observability.md) | Traces/spans/attributes, what to instrument, OpenTelemetry vs. vendor SDKs, cost dashboards, agent-specific alerting, PII in traces |
| 12 | [Security + Guardrails](docs/stage-12-security-guardrails.md) | Prompt injection defense, agentic-specific risks (goal hijacking, memory poisoning, cascading failures), red-teaming, output filtering, PII redaction, sandboxed execution, compliance |
| 13 | [Production Deployment](docs/stage-13-deployment.md) | vLLM/SGLang, Kubernetes scaling, CI/CD for agents, canary releases, idempotent tools, agent-native infra (durable execution, sandboxing, gateways) |
| 14 | [Open Source + Portfolio](docs/stage-14-portfolio.md) | Ship autonomous agents publicly, architecture docs, demos, OSS contributions |

## Notes on sequencing

This path assumes working Python fluency already. If not, insert a Stage 0 covering core syntax, type hints, and virtual environments before Stage 1 — everything after depends on being comfortable reading async stack traces.

**The ordering rule:** stages are sequenced so that every stage depends only on stages before it. Where a stage points *forward* (Stage 2 mentioning Stage 7's long-running agents, Stage 8 mentioning Stage 12's guardrails), that's a "this gets picked up properly later" signpost, never a prerequisite — you can always read a stage having read only its predecessors.

The four load-bearing dependencies, worth knowing because they're the ones that hurt to skip:

| Stage | Genuinely needs | Why |
|---|---|---|
| **07** — Single-agent | 02, 03 | The loop is meaningless without tool calling; the cost intuition comes from 02 |
| **08** — Multi-agent | 06, 07 | Orchestration is single-agent loops sharing sessions. Both halves have to be solid first |
| **10** — Evaluation | 11 (in practice) | Trajectory eval asserts against the spans Stage 11 produces. The numbering says 10 → 11; the build order is usually the reverse |
| **13** — Deployment | 09, 12 | The idempotency and sandboxing sections exist to solve problems those two stages create |

**Two places the numbering and the ideal build order genuinely differ** — noted here rather than renumbered, because the numbers are also a reference index:

- **Stage 11 before Stage 10.** Traces first, then assertions over them. The Beginner Track above already orders it this way.
- **Stage 04–06 after Stage 07, if you're learning alone.** Memory makes far more sense once you have a loop that visibly forgets things. The main sequence puts memory first because that's the dependency order; the Beginner Track puts the loop first because that's the motivation order. Both work — pick based on whether you're building toward a demo or toward completeness.

Frameworks named here (LangGraph, CrewAI, LangSmith, Arize) are one reasonable default stack, not the only correct one — the underlying patterns (supervisor orchestration, tiered memory, eval harnesses, tracing) transfer to any framework or a from-scratch implementation.

## License

Use freely — this is a personal study guide, not licensed software.
