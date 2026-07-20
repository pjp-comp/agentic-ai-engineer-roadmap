[← Stage 04](stage-04-memory-state.md) · Stage 05 of 12 · **Next:** [Stage 06 →](stage-06-multi-agent.md)

# Stage 05 — Single-Agent Workflows

ReAct loops · plan-and-execute · self-reflection · iteration limits · graceful degradation

## Why this matters

Before orchestrating multiple agents, get one agent reliably looping: reason → act → observe → decide, with a hard ceiling on iterations and a defined fallback when it can't converge. This is the load-bearing pattern everything in Stage 6 builds on.

## Brief

Implement a ReAct loop with a max-iteration cap and a stagnation check (same action + same result 2× in a row triggers a forced reflection step, per the Reflexion pattern).

```python
for step in range(MAX_ITERS):
    thought, action = agent.think(state)
    if is_stagnant(action, state.history):
        thought = agent.reflect(state.history)
        action = agent.think(state, hint=thought)
    result = execute(action)
    state.history.append((thought, action, result))
    if action.is_final:
        return result
return degrade_gracefully(state)  # partial answer, not a crash
```

## Sources

| Type | Resource |
|------|----------|
| Paper | [Yao et al. — ReAct: Synergizing Reasoning and Acting (2022)](https://arxiv.org/abs/2210.03629) |
| Paper | [Shinn et al. — Reflexion: Verbal Reinforcement Learning (2023)](https://arxiv.org/pdf/2303.11366) |
| Guide | [ReAct vs Plan-and-Execute vs ReWOO vs Reflexion — compared](https://theaiengineer.substack.com/p/the-4-single-agent-patterns) |
| Docs | [LangGraph — StateGraph / agent loop primitives](https://docs.langchain.com/oss/python/langgraph/overview) |

## Done when

Your agent never runs forever, never crashes on a bad tool result, and returns something useful even when it can't fully solve the task.

---
[← Stage 04 — Memory + State Management](stage-04-memory-state.md) · [Back to roadmap](../README.md) · **Next:** [Stage 06 — Multi-Agent Orchestration →](stage-06-multi-agent.md)
