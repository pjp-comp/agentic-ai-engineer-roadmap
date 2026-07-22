[← Stage 06](stage-06-sessions-state.md) · Stage 07 of 14 · **Next:** [Stage 08 →](stage-08-multi-agent.md)

# Stage 07 — Single-Agent Workflows

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

## Long-running agents

A `for step in range(MAX_ITERS)` loop is fine for a task that finishes in seconds. It's not enough once a task genuinely takes minutes to hours — a large refactor, a multi-source research task, an overnight batch job. Three problems show up that a bounded loop doesn't have to deal with:

1. **The process can die mid-task.** A deploy, an OOM kill, a laptop closing — none of that should mean starting over from step 1.
2. **Cost and context drift as the task grows.** A loop that's been running for 200 steps has a very different token profile than step 5; unbounded context growth eventually blows the window or the budget (this is exactly what Stage 4's short-term/long-term split exists to manage).
3. **"Still working" and "stuck" look identical from outside.** A long-running agent needs to report progress somewhere a human (or a monitor) can check, not just run silently until it returns or times out.

The fix is to make the loop **resumable**, not just boundable: persist enough state after each step that the agent can restart from where it left off instead of from scratch. This is the `PersistentState` piece from [Stage 4](stage-04-memory-state.md) put to use.

```python
def run_long_task(task_id: str, checkpoint_store):
    state = PersistentState(task_id, checkpoint_store)
    saved = state.resume()
    step, history = (saved["step"], saved["history"]) if saved else (0, [])

    while step < MAX_ITERS:
        thought, action = agent.think(history)
        result = execute(action)
        history.append((thought, action, result))
        step += 1

        # Checkpoint every step, not just at the end — this is what makes
        # a kill-and-restart resumable instead of a silent full re-run.
        state.checkpoint({"step": step, "history": history})

        if action.is_final:
            return result

    return degrade_gracefully(history)
```

The difference from the bounded-loop brief above: a crash at step 150 resumes at step 150, not step 0. For anything running unattended past a few minutes, checkpoint on every step, not on a timer — a crash between timer ticks loses everything since the last save.

## Done when — long-running

Killing the process mid-run (`kill -9`) and restarting it resumes from the last checkpoint instead of repeating already-finished work, and you have some way — logs, a status file, a dashboard — to tell "still working" apart from "stuck" without staring at raw output.

---
[← Stage 06 — Sessions, State + Events](stage-06-sessions-state.md) · [Back to roadmap](../README.md) · **Next:** [Stage 08 — Multi-Agent Orchestration →](stage-08-multi-agent.md)
