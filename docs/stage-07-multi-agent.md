[← Stage 06](stage-06-single-agent.md) · Stage 07 of 13 · **Next:** [Stage 08 →](stage-08-human-in-the-loop.md)

# Stage 07 — Multi-Agent Orchestration

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

| Type | Resource |
|------|----------|
| Docs | [LangGraph — Multi-Agent Supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) |
| Repo | [langchain-ai/langgraph-supervisor-py](https://github.com/langchain-ai/langgraph-supervisor-py) |
| Guide | [Supervisor vs. Swarm — tradeoffs and architecture](https://focused.io/lab/multi-agent-orchestration-in-langgraph-supervisor-vs-swarm-tradeoffs-and-architecture) |
| Course | [Ed Donner — AI Engineer Agentic Track (5 frameworks, 8 projects)](https://www.udemy.com/course/the-complete-agentic-ai-engineering-course/) |
| Framework | CrewAI (role-based crews) as a contrast to LangGraph's explicit graph model |

## Done when

You can justify, with your own token/latency numbers, whether a task actually needed multiple agents or would've been cheaper as one.

---
[← Stage 06 — Single-Agent Workflows](stage-06-single-agent.md) · [Back to roadmap](../README.md) · **Next:** [Stage 08 — Human-in-the-Loop Systems →](stage-08-human-in-the-loop.md)
