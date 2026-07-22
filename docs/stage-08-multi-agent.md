[← Stage 07](stage-07-single-agent.md) · Stage 08 of 14 · **Next:** [Stage 09 →](stage-09-human-in-the-loop.md)

# Stage 08 — Multi-Agent Orchestration

LangGraph/CrewAI · supervisor patterns · message passing · conflict resolution · handoffs

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

## Sources

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
