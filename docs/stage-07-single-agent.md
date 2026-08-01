[← Stage 06](stage-06-sessions-state.md) · Stage 07 of 14 · **Next:** [Stage 08 →](stage-08-multi-agent.md)

# Stage 07 — Single-Agent Workflows

Workflow patterns · ReAct loops · Tree-of-Thought · self-reflection · iteration limits · graceful degradation · cognitive architecture (confidence gating, attention, knowledge boundaries) · harness/loop/graph engineering

## Why this matters

Before orchestrating multiple agents, get one agent reliably looping: reason → act → observe → decide, with a hard ceiling on iterations and a defined fallback when it can't converge. This is the load-bearing pattern everything in Stage 8 builds on.

## Harness, Loop, and Graph — the current framing for how agents get built

By 2026 the field settled on a three-part vocabulary for what "building an agent" actually means, popularized by Martin Fowler's writing on coding-agent harnesses and Vin Vashishta's "Harness, Loop, and Graph" framework. It's worth knowing by name because it's a map of this entire roadmap, not a new technique — it names things you've already been building in Stages 2–8 and clarifies which layer a given problem actually lives in.

**Agent = Model + Harness.** The model reasons; the harness is everything else — the code, config, and execution logic that turns a model's raw text output into governed, auditable action. Three questions, three layers:

| Layer | Answers | Where it lives in this roadmap |
|---|---|---|
| **Harness** | What can the agent see and do? | Stage 2 (context engineering, what goes into the window), Stage 3 (tool schemas, what actions are possible), Stage 4 (memory — what persists), Stage 6 (session/state — what's in scope this turn), Stage 12 (permissions, sandboxing, what's allowed) |
| **Loop** | How does it decide what to do next, and when does it stop? | This stage's ReAct loop, Reflexion/stagnation check, iteration caps, and graceful degradation — all above and below this section |
| **Graph** | Does one agent do everything, or is the work split across several? | Stage 8 — supervisor patterns, handoffs, multi-agent topology |

**The differentiation that actually matters:** these aren't competing architectures you choose between — they're three different questions about the *same* system, and conflating them is the most common source of confusion when reading about "agent architecture" online. A weak harness (a tool with a vague schema, a context window stuffed with irrelevant history) makes the loop unreliable no matter how well the loop itself is designed — the model is reasoning well over bad inputs. A well-designed loop with no graph is still just one agent; adding a graph doesn't fix a loop that doesn't know when to stop, it just runs that same unreliable loop in more places at once. Fixing the wrong layer is a common debugging trap: an agent that "hallucinates" a tool call that doesn't exist is usually a harness problem (bad tool schema or missing context), not a loop problem — no amount of retry logic fixes a tool the model was never given accurate information about.

**Practical ordering:** get the harness right first (Stages 2–6), then the loop (this stage), then reach for a graph (Stage 8) only once a single well-harnessed, well-looped agent is the actual bottleneck — not before. This is the same "start simple" advice the six-workflow-patterns table below gives, just phrased at the level of the system's three structural layers instead of one call's control flow.

## Beginner focus

- Build one simple ReAct loop first.
- Add a max-iteration limit.
- Add a graceful fallback response.
- Skip advanced workflow variants on your first build.

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
| Article | [Martin Fowler — Harness engineering for coding agent users](https://martinfowler.spicytakes.org/post/2026-04-02-harness-engineering) — source for the harness/loop/graph section above |
| Guide | [Vin Vashishta — Harness, Loop, & Graph: A Simple Explanation of How AI Agents Are Built](https://vinvashishta.substack.com/p/harness-loop-and-graph-a-simple-explanation) |
| Guide | [Vin Vashishta — Harness, Loop, & Graph, Part 2](https://vinvashishta.substack.com/p/harness-loop-and-graph-part-2-a-simple) |
| Blog | [Databricks — What is an AI Agent Harness?](https://www.databricks.com/blog/ai-harness) |
| Book | Michael Yuan — *AI Agents in Action*, 2nd ed. (Manning) — source for Tree-of-Thought and the cognitive-architecture (perception/planning/execution/evaluation/attention) framing below; [repo with runnable examples](https://github.com/cxbxmxcx/AI-Agent-Workflows) |

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
| **Tree-of-Thought (ToT)** | Explores multiple reasoning branches at once, evaluates each, backtracks from dead ends | A single reasoning chain (plain CoT/ReAct) is prone to committing early to a wrong path and a problem has a search-like structure (puzzles, multi-step planning with backtracking) — costs multiples of a single ReAct pass, since you're running several branches |

Anthropic's own framing is worth internalizing verbatim: **start with the simplest pattern (or no framework at all) and add complexity only when it demonstrably improves outcomes.** A prompt chain that works is better than an autonomous agent that's harder to debug for the same result. This stage teaches the most powerful pattern in the table, not the default one — pick it because the task needs it, not because it's the most sophisticated option available.

## Beyond request/response: computer-use and voice agents

Everything above assumes a text-in, text-out loop — the model reasons, calls a tool, gets a result back, and eventually returns text. Two 2025–2026 categories break that assumption and are worth knowing exist, even if this roadmap doesn't build them:

- **Computer-use / browser-use agents** — instead of calling an API-shaped tool, the model perceives a screenshot and issues GUI actions (click, type, scroll). This is a different tool paradigm from Stage 3's function-calling, not a variant of it — see [Claude's computer use tool](https://platform.claude.com/docs/en/agents-and-tools/computer-use/overview) and the [Claude vs. OpenAI vs. Gemini computer-use comparison](https://www.digitalapplied.com/blog/computer-use-agents-2026-claude-openai-gemini-matrix) if a task genuinely has no API and a UI is the only integration point.
- **Voice/realtime agents** — a persistent streaming connection (WebSocket/WebRTC) instead of discrete request/response calls, with sub-500ms latency budgets and interruption ("barge-in") handling. This is architecturally distinct enough from the ReAct loop in this stage that it's effectively a separate track — see the [voice-agent orchestration tooling overview](https://www.assemblyai.com/blog/orchestration-tools-ai-voice-agents) if that's the direction a project needs.

Both are legitimate production categories now, not research demos — they're flagged here rather than built out because they change the I/O model this entire roadmap is built around, and are worth a dedicated project once the text-based patterns above are solid.

## From reasoning primitives to a cognitive architecture

Everything above — ReAct, Reflexion, ToT, the workflow-pattern table — are reasoning *primitives*: techniques you reach for per call or per loop. A **cognitive architecture** is a step up from that: a fixed set of modules the agent runs through every cycle, so "thinking" isn't just prompting technique but a repeatable structure. The five modules worth knowing, from the "cognitive agent" pattern popularized by Michael Yuan's *AI Agents in Action* (2nd ed., Manning):

| Module | Job | How it differs from what's already in this roadmap |
|---|---|---|
| **Perception** | Reads the current state and incoming input | Same as "Perceive" in the ReAct loop above — no new concept |
| **Planning** | Decides the next step or sub-goal | Same as "Decide" — this stage's workflow-pattern table already covers the options |
| **Execution** | Runs the chosen action | Same as "Act" — Stage 3's tool calling |
| **Evaluation** | Scores its own output against the goal, every cycle — not just when stuck | **New**: this stage's stagnation check only reflects when a loop is visibly stuck (same action/result twice). A dedicated evaluation module runs every cycle, catching a plan that's technically progressing but drifting off-goal, which a stagnation check would never trigger on |
| **Attention** | Decides what part of the accumulated context is relevant *right now*, and deliberately ignores the rest | **New**: distinct from Stage 2's context budgeting (which prunes for token cost) — attention filters for *relevance*, not size; a context window can be well within budget and still have the model reasoning over stale or irrelevant history because nothing tells it what to ignore |

Two behaviors this architecture adds that pure ReAct+Reflexion doesn't have a name for:

- **Confidence-gated execution** — before acting, the agent scores its own confidence in the plan; below a threshold, it seeks more information (asks a clarifying question, does another retrieval pass) instead of acting on a low-confidence guess. This is stricter than Reflexion, which only intervenes after an action visibly fails or repeats — confidence gating can stop a bad action *before* it runs once.
- **Knowledge-boundary awareness** — the agent explicitly represents what it doesn't know (distinct from Stage 5's "don't trust weak retrieval" check in CRAG), so it can say "I don't have enough information" as a first-class outcome rather than only reaching that conclusion after retrieval comes up empty.

Whether this is worth building as actual separate modules or just borrowing the *vocabulary* (evaluation-every-cycle, attention-as-filtering, confidence gating) to sharpen an existing ReAct+Reflexion loop is a judgment call — for most single-agent projects, folding confidence-gating and an explicit "what am I unsure about" check into the existing loop gets most of the benefit without the full module architecture.

---
[← Stage 06 — Sessions, State + Events](stage-06-sessions-state.md) · [Back to roadmap](../README.md) · **Next:** [Stage 08 — Multi-Agent Orchestration →](stage-08-multi-agent.md)
