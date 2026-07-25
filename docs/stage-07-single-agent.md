[← Stage 06](stage-06-sessions-state.md) · Stage 07 of 14 · **Next:** [Stage 08 →](stage-08-multi-agent.md)

# Stage 07 — Single-Agent Workflows

Workflow patterns · ReAct loops · plan-and-execute · self-reflection · iteration limits · graceful degradation

## Why this matters

Before orchestrating multiple agents, get one agent reliably looping: reason → act → observe → decide, with a hard ceiling on iterations and a defined fallback when it can't converge. This is the load-bearing pattern everything in Stage 8 builds on.

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
| Guide | [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — the source for the six-pattern table below; read this before reaching for the most complex pattern by default |
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

## The six workflow patterns — where ReAct fits among them

The ReAct loop above is one of six named patterns from Anthropic's ["Building Effective Agents"](https://www.anthropic.com/engineering/building-effective-agents) — the field's most-cited reference on this. Knowing all six matters because the biggest mistake at this stage isn't picking the wrong one, it's reaching for autonomous ReAct (the most complex, most expensive) when a much simpler pattern would do:

| Pattern | Shape | Reach for it when |
|---|---|---|
| **Prompt chaining** | Fixed sequence of LLM calls, each step's output feeding the next | The task decomposes into a known, unchanging sequence of subtasks — no dynamic decision-making needed |
| **Routing** | One call classifies, then dispatches to a specialized path | Inputs fall into distinct categories that genuinely need different handling (this is Stage 2's router) |
| **Parallelization** | Multiple LLM calls run concurrently, then results are combined or voted on | Independent subtasks can run at once (sectioning), or you want multiple attempts scored against each other (voting) for higher confidence |
| **Orchestrator-workers** | A central call breaks work into dynamic subtasks, dispatches to workers, synthesizes results | The subtasks aren't known in advance — this is Stage 8's supervisor pattern |
| **Evaluator-optimizer** | One call generates, a second call critiques against explicit criteria, loop until it passes | There's a clear, checkable quality bar and iterative refinement measurably improves the result — distinct from this stage's Reflexion/stagnation check, which reacts to a *stuck* loop rather than running a dedicated critic every pass |
| **Autonomous agents (ReAct)** | Open-ended loop: the model decides its own steps until done | The path to the answer genuinely can't be predetermined — this stage's pattern, and the most expensive one to get wrong |

Anthropic's own framing is worth internalizing verbatim: **start with the simplest pattern (or no framework at all) and add complexity only when it demonstrably improves outcomes.** A prompt chain that works is better than an autonomous agent that's harder to debug for the same result. This stage teaches the most powerful pattern in the table, not the default one — pick it because the task needs it, not because it's the most sophisticated option available.

## Beyond request/response: computer-use and voice agents

Everything above assumes a text-in, text-out loop — the model reasons, calls a tool, gets a result back, and eventually returns text. Two 2025–2026 categories break that assumption and are worth knowing exist, even if this roadmap doesn't build them:

- **Computer-use / browser-use agents** — instead of calling an API-shaped tool, the model perceives a screenshot and issues GUI actions (click, type, scroll). This is a different tool paradigm from Stage 3's function-calling, not a variant of it — see [Claude's computer use tool](https://platform.claude.com/docs/en/agents-and-tools/computer-use/overview) and the [Claude vs. OpenAI vs. Gemini computer-use comparison](https://www.digitalapplied.com/blog/computer-use-agents-2026-claude-openai-gemini-matrix) if a task genuinely has no API and a UI is the only integration point.
- **Voice/realtime agents** — a persistent streaming connection (WebSocket/WebRTC) instead of discrete request/response calls, with sub-500ms latency budgets and interruption ("barge-in") handling. This is architecturally distinct enough from the ReAct loop in this stage that it's effectively a separate track — see the [voice-agent orchestration tooling overview](https://www.assemblyai.com/blog/orchestration-tools-ai-voice-agents) if that's the direction a project needs.

Both are legitimate production categories now, not research demos — they're flagged here rather than built out because they change the I/O model this entire roadmap is built around, and are worth a dedicated project once the text-based patterns above are solid.

---
[← Stage 06 — Sessions, State + Events](stage-06-sessions-state.md) · [Back to roadmap](../README.md) · **Next:** [Stage 08 — Multi-Agent Orchestration →](stage-08-multi-agent.md)
