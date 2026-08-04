[← Stage 07](stage-07-single-agent.md) · Stage 08 of 14 · **Next:** [Stage 09 →](stage-09-human-in-the-loop.md)

# Stage 08 — Multi-Agent Orchestration

Framework landscape (LangGraph/CrewAI/ADK/OpenAI Agents SDK/Pydantic AI/Mastra) · graph engineering (topology, node types, failure isolation) · supervisor patterns · message passing · conflict resolution · handoffs + inline guardrails · A2A cross-agent protocol + AAIF governance · adjacent commerce protocols (AP2/x402/UCP)

## Why this matters

Splitting work across specialized agents buys focused prompts and parallelism, but costs roughly 3× the tokens of one mega-agent for the coordination overhead — so it's a tool for genuinely decomposable tasks, not a default. Start with the supervisor pattern: one router LLM, deterministic Python for the actual routing logic.

This stage is the **graph** layer in the harness/loop/graph framework introduced in [Stage 7](stage-07-single-agent.md#harness-loop-and-graph--the-current-framing-for-how-agents-get-built) — deciding whether one agent handles everything or several split the work. Reach for it only after Stage 7's harness and loop are solid for a single agent; a graph doesn't fix an unreliable loop, it just runs that same unreliable loop in more places at once.

## Brief

Build a supervisor with two worker agents (e.g. a researcher and a writer) as a LangGraph `StateGraph`. The supervisor node decides which worker runs next based on shared state; workers write results back to state, never talk to each other directly.

```python
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_conditional_edges(
    "supervisor", route_next,
    {"researcher": "researcher", "writer": "writer", "done": END},
)
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
```

**New to `StateGraph`?** [`examples/stage03-tool-calling-langgraph/`](../examples/stage03-tool-calling-langgraph/) builds the smallest possible one first — a single-node loop, not a supervisor — by rebuilding example 01's raw-API agent on LangGraph, with a line-by-line map of what `StateGraph`, `ToolNode`, and `tools_condition` each replace. Worth running before jumping straight to a two-worker supervisor graph.

## Beyond LangGraph/CrewAI — the wider 2026 framework field

LangGraph and CrewAI are one reasonable default, not the whole field. Knowing what else exists matters when a project's constraints (language, team preference, or a specific pattern the framework makes easy) point elsewhere:

| Framework | Language | Fits when |
|---|---|---|
| **LangGraph** | Python/JS | You want the explicit graph model this stage teaches — nodes, edges, and state you control directly |
| **CrewAI** | Python | Role-based "crews" (a more opinionated, less explicit alternative to LangGraph's graph) |
| **Google Agent Development Kit (ADK)** | Python/Java | Already cited in [Stage 6](stage-06-sessions-state.md) for its session/state/event model — also a full orchestration framework in its own right, with first-class multi-agent support |
| **OpenAI Agents SDK** | Python/JS | The production successor to OpenAI's earlier Swarm experiment — built-in agent-as-tool and handoff patterns, minimal abstraction over raw API calls |
| **Pydantic AI** | Python | Type-safe agent definitions via Pydantic models — increasingly the pick for single-agent-with-tools work where LangGraph's graph machinery is more than the task needs |
| **Mastra** | TypeScript | The de facto default for TypeScript/Node agent projects, filling the role LangGraph/CrewAI play in Python |

None of these change the underlying patterns from Stage 7 (routing, orchestrator-workers, evaluator-optimizer, ReAct) — they're different amounts of scaffolding around the same ideas. Picking one is a team/language/opinionatedness decision, not a capability decision.

## A2A — agent-to-agent, across vendors and processes

Stage 6's shared-session pattern is how two agents communicate *inside one process, one framework*. That doesn't extend to agents built by different teams on different frameworks that need to talk to each other — a supervisor written in LangGraph calling a specialist agent someone else built in CrewAI, for instance. **A2A (Agent2Agent)**, introduced by Google in April 2025 and donated to the Linux Foundation two months later, is the emerging standard for exactly that: each agent publishes an "Agent Card" describing its capabilities, and agents negotiate and exchange tasks over a defined protocol — independent of what either side is built with. IBM's separately-developed Agent Communication Protocol (also, confusingly, abbreviated ACP) merged into A2A in August 2025 rather than competing with it, and A2A's 1.0 release added **signed Agent Cards** (JWS/RFC 7515) so an agent's published capabilities can be cryptographically verified, not just trusted at face value.

This is a different layer from MCP: MCP is agent↔tool (Stage 3); A2A is agent↔agent, across process and vendor boundaries. For a single-framework project like the supervisor pattern in this stage's brief, you don't need it — shared session state is simpler and sufficient. Reach for A2A only when a project genuinely needs to interoperate with an agent it doesn't control the framework for.

**Governance, as of December 2025:** MCP and A2A are no longer single-vendor projects. Anthropic donated MCP to a new **Agentic AI Foundation (AAIF)**, a directed fund under the Linux Foundation co-founded with Block and OpenAI, with Google, Microsoft, AWS, Cloudflare, and Bloomberg among the supporting members — the explicit goal being a neutral home so no single company controls the protocol's direction. A2A had already moved to Linux Foundation governance in June 2025. Practically: when picking a protocol to build on, "which company owns this" is less of a lock-in risk for MCP/A2A now than it would have been in early 2025 — worth knowing if you're choosing between a foundation-governed standard and a single-vendor one for anything you plan to depend on long-term.

## Adjacent protocols: agent commerce (know these exist, don't build on them yet)

MCP (tools) and A2A (agent-to-agent) aren't the whole emerging protocol landscape — a newer, less mature layer covers agents *transacting*, not just communicating. Three real, distinct efforts, worth knowing apart because their names collide:

- **AP2 (Agent Payments Protocol)** — announced by Google in September 2025 with 60+ launch partners (including Coinbase); covers an agent completing a payment on a user's behalf with verifiable authorization.
- **x402** — Coinbase's HTTP-402-based stablecoin payment scheme, now under Linux Foundation governance; lets an API respond "payment required" and an agent complete that payment as part of the request flow.
- **UCP (Universal Commerce Protocol)** — Google's protocol for agent-driven shopping/commerce, backed by Shopify, Target, Walmart, Mastercard, and Visa.

**A naming collision worth knowing, not just a footnote:** there are two unrelated protocols both abbreviated "ACP." IBM's **Agent Communication Protocol** — the one that merged into A2A above — is general agent-to-agent messaging. A **separate, unrelated "Agentic Commerce Protocol,"** also abbreviated ACP, was created by **OpenAI and Stripe** specifically for commerce, not IBM or the Linux Foundation. If you see "ACP" in a commerce context, it's almost certainly this second one, not the one that merged into A2A.

None of this is mature enough to be a "learn this next" recommendation the way MCP and A2A are — treat it as a landscape to be aware of, not something to build a project on yet. If Stage 12's threat modeling ever needs to cover agent payments specifically, this is the section that would grow.

## Graph engineering, in depth — why the topology is a first-class design decision

Stage 7 named "graph" as one of three structural layers (harness/loop/graph) and pointed here. This section is the actual depth: why the shape of a multi-agent system is something you design deliberately, not something that falls out naturally from "add more agents." Treat this as an emerging framing, not a settled standard — the vocabulary below (nodes, node failure isolation, dynamic node spawning) is still being formalized across the field in 2026, and different sources use overlapping but not identical terms for the same ideas.

**Why it's required, not optional, past a certain point.** A two-agent supervisor (this stage's brief) barely needs "graph engineering" as a discipline — the topology is obvious, there's one router and two workers. The discipline earns its name once a system has enough nodes that the topology itself becomes a source of bugs: which node can call which, what happens when a node fails mid-graph, whether two branches can run concurrently and race on shared state. At that point the graph's *shape* is doing real architectural work, and treating it as an afterthought (just wire nodes together until it works) is where most of Stage 8's "3× token cost for coordination overhead" warning actually comes from — a badly-shaped graph doesn't just cost more, it fails in ways that are hard to reproduce.

**What counts as a node.** Not every node is an agent. A real work graph mixes:

| Node type | What it does | Example in this stage's brief |
|---|---|---|
| **Agent** | An LLM reasoning loop (Stage 7's harness+loop) | `researcher_node`, `writer_node` |
| **Router** | Deterministic code choosing the next node — no LLM call | `route_next` in this stage's brief |
| **Tool/function** | A plain function call, no reasoning | Stage 3's tool-calling, wrapped as a graph node instead of called inline |
| **Join** | Waits for multiple branches to complete before proceeding | Not in this stage's brief (which is sequential); needed once you parallelize (Stage 7's "parallelization" workflow pattern, expressed as a graph) |
| **Human checkpoint** | Pauses the graph for approval | Stage 9's human-in-the-loop pattern, expressed as a graph node |

Mixing these deliberately — not making every node an expensive LLM call — is itself a design decision: a router or a join doesn't need to reason, so making it a plain function instead of another agent is real cost savings, not a missed opportunity for "more AI."

**How to actually use it — the design questions, in order:**

1. **Which nodes exist, and what does each one own?** Answer this before writing any code — "researcher owns finding facts, writer owns prose, nothing else touches either's output" is a topology decision, not an implementation detail.
2. **Which transitions are permitted?** Not every node should be able to route to every other node — an explicit allowed-transitions list (this stage's `add_conditional_edges` mapping) is what prevents a routing bug from silently sending work somewhere it was never designed to go.
3. **Where do joins and human checkpoints sit?** Decide up front whether any two nodes can run concurrently (and therefore need a join before their results are combined), and whether any transition needs a human gate (Stage 9) before proceeding — retrofitting either into an existing graph is markedly harder than designing them in.
4. **How does the graph observe itself?** Stage 11's tracing needs stable identifiers per graph, per run, and per node — not just per LLM call — so a failure can be attributed to *which node in which run*, not just "the agent did something wrong" somewhere in a long trace.

**Where it actually breaks — the difficulties worth knowing before you hit them:**

- **Garbage in, garbage forwarded.** A graph doesn't fix a bad node — if a node starts with wrong information or finishes without a way to verify its own output, the graph just moves that mistake to the next node instead of catching it. This is the same point Stage 8's existing "Guardrails at the handoff" section makes below, restated at the topology level: the graph's edges are exactly where validation belongs, because that's where a bad result would otherwise propagate unchecked.
- **Node failure isolation.** In a single-agent loop, a crash is the whole program crashing. In a graph, one node failing shouldn't silently corrupt shared state or leave the graph in an inconsistent position — this needs the same resumability thinking as Stage 7's long-running-agent section, applied per-node instead of per-loop-step.
- **State consistency across concurrent branches.** Once two nodes can run in parallel (a join, not just a sequential chain), they can race on shared state the same way concurrent writes to a database can — Stage 6's session/state model needs the same care here that a multi-writer system needs anywhere else in software.
- **The harness gets harder, not easier.** A graph's harness (Stage 7's harness layer, applied to the whole graph instead of one agent) starts to resemble a distributed-systems runtime — routing, retries, timeouts, per-node budgets and permissions — not just a single agent's tool list and context window. This is why production multi-agent platforms (LangGraph Platform, TrueFoundry's Agent Gateway, and similar) increasingly treat inter-agent traffic as regulated API traffic with per-node identity and budgets, rather than trusting the graph to self-regulate.

**The practical takeaway**: design the topology on paper (which nodes, which transitions, where joins and human checkpoints go) before writing the graph in code — the same "smallest brief that forces you to touch the real API surface" philosophy this roadmap uses everywhere else applies here too. A two-node supervisor doesn't need this rigor; a graph with five or more heterogeneous node types does.

## Guardrails at the handoff, not just at the end

[Stage 12](stage-12-security-guardrails.md) covers guardrails as a dedicated security pass — but in a multi-agent flow, validation belongs at every handoff, not only as a final check before output leaves the system. Two failure modes this stage's supervisor pattern is exposed to if handoffs go unvalidated:

- **Bad input propagates unchecked.** If the researcher writes a malformed or off-topic result to shared state, the writer will draft from it in good faith — the shared-session model (Stage 6) has no built-in notion of "don't trust this handoff," so a bad result compounds instead of getting caught at the boundary where it originated.
- **One agent validating another inline** — a lightweight pattern worth naming: a small "grounding" or "critic" check runs *between* two worker nodes (e.g., between `researcher` and `writer` in this stage's brief), rejecting or routing back a handoff that fails a basic check (empty result, off-topic, missing required fields), rather than only catching it in Stage 10's eval suite after the fact or Stage 12's guardrails at the system boundary.

This isn't a replacement for Stage 12's guardrails (prompt injection, PII, sandboxing still apply at the system's edges) — it's a narrower, cheaper check that belongs in the graph itself: a conditional edge that routes a failing handoff back to the sender instead of forward to the next worker, the same shape as `route_next` in this stage's brief but validating content, not just deciding "who's next."

## Sources

### Graph engineering

| Type | Resource |
|------|----------|
| Guide | [TrueFoundry — Graph Engineering for Multi-Agent Systems: Architecture, Governance, and Observability](https://www.truefoundry.com/blog/graph-engineering-enterprise-guide) — source for node types, node failure isolation, and gateway-first governance above |
| Guide | [explainx.ai — Graph Engineering: Wire Multi-Agent Orgs After Loops](https://www.explainx.ai/blog/graph-engineering-ai-agents-multi-agent-organizations-2026) |
| Guide | [Eigent — Graph Engineering for AI Agents](https://www.eigent.ai/blog/graph-engineering-ai-agents) |

### Framework landscape and A2A

| Type | Resource |
|------|----------|
| Guide | [LangChain — Best AI agent frameworks 2026](https://www.langchain.com/resources/ai-agent-frameworks) |
| Guide | [morphllm — 8 agent SDKs compared](https://www.morphllm.com/ai-agent-framework) |
| Docs | [Google Agent Development Kit](https://google.github.io/adk-docs/) |
| Docs | [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) |
| Spec | [A2A Protocol specification](https://a2a-protocol.org/latest/specification/) |
| Announcement | [A2A 1.0 — signed Agent Cards, cryptographic verification](https://a2a-protocol.org/latest/announcing-1.0/) |
| Announcement | [Google — A2A, one year of open collaboration](https://opensource.googleblog.com/2026/04/a-year-of-open-collaboration-celebrating-the-anniversary-of-a2a.html) |
| Announcement | [Linux Foundation — IBM's ACP joins forces with A2A](https://lfaidata.foundation/communityblog/2025/08/29/acp-joins-forces-with-a2a-under-the-linux-foundations-lf-ai-data/) |
| Announcement | [Anthropic — Donating MCP, establishing the Agentic AI Foundation](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) |
| Guide | [A2A in 2026 — Adoption, Hype, Reality](https://www.glukhov.org/ai-systems/comparisons/a2a-protocol-2026-adoption/) |
| Announcement | [Google — Universal Commerce Protocol (UCP)](https://developers.googleblog.com/under-the-hood-universal-commerce-protocol-ucp/) |
| Announcement | [Google Cloud — AP2 (Agent Payments Protocol)](https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol) |
| Announcement | [Coinbase — x402 launch](https://www.coinbase.com/developer-platform/discover/launches/x402) |
| Spec | [OpenAI/Stripe — Agentic Commerce Protocol (the *other* ACP)](https://github.com/agentic-commerce-protocol/agentic-commerce-protocol) |
| Book | Michael Yuan — *AI Agents in Action*, 2nd ed. (Manning), ch. 4 — source for guardrails-at-handoff and agents-as-guardrails, above; [repo with runnable examples](https://github.com/cxbxmxcx/AI-Agent-Workflows) |

### Learn LangGraph itself (official, start here)

| Type | Resource |
|------|----------|
| Course | [LangChain Academy](https://academy.langchain.com/) — free, from the team that builds LangGraph; modules 1–5 build up the framework, module 6 covers deployment |
| Repo | [langchain-ai/langchain-academy](https://github.com/langchain-ai/langchain-academy) — the course notebooks, runnable locally |
| Docs | [LangChain — Learn section](https://docs.langchain.com/oss/python/learn) — official tutorials and conceptual overviews |
| Docs | [LangGraph — Multi-Agent Supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) |
| Repo | [langchain-ai/langgraph-supervisor-py](https://github.com/langchain-ai/langgraph-supervisor-py) |
| Guide | [Supervisor vs. Swarm — tradeoffs and architecture](https://focused.io/lab/multi-agent-orchestration-in-langgraph-supervisor-vs-swarm-tradeoffs-and-architecture) |
| Course | [Ed Donner — AI Engineer Agentic Track (5 frameworks, 8 projects)](https://www.udemy.com/course/the-complete-agentic-ai-engineering-course/) |
| Framework | CrewAI (role-based crews) as a contrast to LangGraph's explicit graph model |

### Stock/financial analysis multi-agent examples (directly applicable)

| Type | Resource |
|------|----------|
| Repo | [PrimoAgent — multi-agent stock analysis](https://github.com/ivebotunac/PrimoAgent) — four specialized agents (data collection, technical analysis, news intelligence, portfolio manager) orchestrated via LangGraph; close match for the researcher/analyst/sentiment/risk split described earlier in this roadmap |
| Repo | [Multi-agent finance analysis with LangGraph](https://github.com/AI-lab-sh/Multi-agent-finance-analysis-by-langgraph) — recommendation system pulling from multiple online sources |
| Tutorial | [Building a Multi-Agent Stock Research Assistant with LangGraph + Gemini](https://medium.com/@christopher-thomas/building-a-multi-agent-stock-research-assistant-with-langgraph-and-google-gemini-54e0249a3a86) |
| Tutorial | [LangGraph + MCP: Build a stock analysis agent](https://medium.com/@sitabjapal03/langgraph-mcp-build-a-stock-analysis-agent-part-1-34bbf431610d) — ties together Stage 3's MCP section with this stage's supervisor pattern |
| Course | [DataCamp — Multi-Agent Systems with LangGraph](https://www.datacamp.com/courses/multi-agent-systems-with-langgraph) — builds a Fortune 500 stock-performance assistant from single-agent up to a three-agent supervisor |
| Reference | [Multi-agent hedge fund simulation](https://shaikhmubin.medium.com/multi-agent-hedge-fund-simulation-with-langchain-and-langgraph-64060aabe711) — portfolio manager + fundamental/technical/sentiment analyst agents, the same role split most stock-analysis projects converge on |

## Done when

You can justify, with your own token/latency numbers, whether a task actually needed multiple agents or would've been cheaper as one.

---
[← Stage 07 — Single-Agent Workflows](stage-07-single-agent.md) · [Back to roadmap](../README.md) · **Next:** [Stage 09 — Human-in-the-Loop Systems →](stage-09-human-in-the-loop.md)
