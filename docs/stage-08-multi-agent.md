[← Stage 07](stage-07-single-agent.md) · Stage 08 of 14 · **Next:** [Stage 09 →](stage-09-human-in-the-loop.md)

# Stage 08 — Multi-Agent Orchestration

Framework landscape (LangGraph/CrewAI/ADK/OpenAI Agents SDK/Pydantic AI/Mastra) · supervisor patterns · message passing · conflict resolution · handoffs + inline guardrails · A2A cross-agent protocol

## Why this matters

Splitting work across specialized agents buys focused prompts and parallelism, but costs roughly 3× the tokens of one mega-agent for the coordination overhead — so it's a tool for genuinely decomposable tasks, not a default. Start with the supervisor pattern: one router LLM, deterministic Python for the actual routing logic.

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

**New to `StateGraph`?** [`examples/01-basic-agent-langgraph/`](../examples/01-basic-agent-langgraph/) builds the smallest possible one first — a single-node loop, not a supervisor — by rebuilding example 01's raw-API agent on LangGraph, with a line-by-line map of what `StateGraph`, `ToolNode`, and `tools_condition` each replace. Worth running before jumping straight to a two-worker supervisor graph.

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

Stage 6's shared-session pattern is how two agents communicate *inside one process, one framework*. That doesn't extend to agents built by different teams on different frameworks that need to talk to each other — a supervisor written in LangGraph calling a specialist agent someone else built in CrewAI, for instance. **A2A (Agent2Agent)**, now a Linux Foundation project with broad industry backing, is the emerging standard for exactly that: each agent publishes an "Agent Card" describing its capabilities, and agents negotiate and exchange tasks over a defined protocol — independent of what either side is built with.

This is a different layer from MCP: MCP is agent↔tool (Stage 3); A2A is agent↔agent, across process and vendor boundaries. For a single-framework project like the supervisor pattern in this stage's brief, you don't need it — shared session state is simpler and sufficient. Reach for A2A only when a project genuinely needs to interoperate with an agent it doesn't control the framework for.

## Guardrails at the handoff, not just at the end

[Stage 12](stage-12-security-guardrails.md) covers guardrails as a dedicated security pass — but in a multi-agent flow, validation belongs at every handoff, not only as a final check before output leaves the system. Two failure modes this stage's supervisor pattern is exposed to if handoffs go unvalidated:

- **Bad input propagates unchecked.** If the researcher writes a malformed or off-topic result to shared state, the writer will draft from it in good faith — the shared-session model (Stage 6) has no built-in notion of "don't trust this handoff," so a bad result compounds instead of getting caught at the boundary where it originated.
- **One agent validating another inline** — a lightweight pattern worth naming: a small "grounding" or "critic" check runs *between* two worker nodes (e.g., between `researcher` and `writer` in this stage's brief), rejecting or routing back a handoff that fails a basic check (empty result, off-topic, missing required fields), rather than only catching it in Stage 10's eval suite after the fact or Stage 12's guardrails at the system boundary.

This isn't a replacement for Stage 12's guardrails (prompt injection, PII, sandboxing still apply at the system's edges) — it's a narrower, cheaper check that belongs in the graph itself: a conditional edge that routes a failing handoff back to the sender instead of forward to the next worker, the same shape as `route_next` in this stage's brief but validating content, not just deciding "who's next."

## Sources

### Framework landscape and A2A

| Type | Resource |
|------|----------|
| Guide | [LangChain — Best AI agent frameworks 2026](https://www.langchain.com/resources/ai-agent-frameworks) |
| Guide | [morphllm — 8 agent SDKs compared](https://www.morphllm.com/ai-agent-framework) |
| Docs | [Google Agent Development Kit](https://google.github.io/adk-docs/) |
| Docs | [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) |
| Spec | [A2A Protocol specification](https://a2a-protocol.org/latest/specification/) |
| Guide | [A2A in 2026 — Adoption, Hype, Reality](https://www.glukhov.org/ai-systems/comparisons/a2a-protocol-2026-adoption/) |
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
