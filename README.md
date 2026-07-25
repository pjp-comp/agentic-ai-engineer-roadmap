# Becoming an Agentic AI Engineer

A sequenced, 14-stage curriculum from async Python fundamentals to shipping observable, guarded, production multi-agent systems. Each stage lists what to build, a runnable code brief, a completion check, and vetted sources — official docs first, courses and papers second.

- **Prepared:** 2026-07-17
- **Pace:** ~4–7 months at 8–10 hrs/week

## How to use this

Stages are ordered by dependency, not difficulty — Stage 8 assumes Stages 3, 6, and 7 are solid, since orchestration is just single-agent loops sharing sessions. Don't skip the "brief" in each stage: it's the smallest program that forces you to touch the real API surface, not a toy print statement.

Treat official docs as ground truth and everything else (courses, blog tutorials) as commentary on it — frameworks like LangGraph ship breaking changes often enough that an older tutorial can mislead you on current APIs.

## Beginner Track (Simple)

If you are a beginner, follow this path first and keep the goal narrow: build one stable single-agent application before adding advanced architecture.

Beginner order:

1. [Stage 01 — Python + Async Foundations](docs/stage-01-python-async.md)
2. [Stage 02 — LLM Fundamentals for Agents](docs/stage-02-llm-fundamentals.md)
3. [Stage 03 — Tool Calling + Structured Outputs](docs/stage-03-tool-calling.md)
4. [Stage 04 — Memory + State Management](docs/stage-04-memory-state.md)
5. [Stage 06 — Sessions, State + Events](docs/stage-06-sessions-state.md)
6. [Stage 07 — Single-Agent Workflows](docs/stage-07-single-agent.md)
7. [Stage 10 — Evaluation + Quality Assurance](docs/stage-10-evaluation-qa.md)
8. [Stage 11 — Observability + Tracing](docs/stage-11-observability.md)
9. [Stage 12 — Security + Guardrails](docs/stage-12-security-guardrails.md)
10. [Stage 13 — Production Deployment](docs/stage-13-deployment.md)

Optional later (after your first stable project):

- [Stage 05 — RAG + Retrieval](docs/stage-05-rag-retrieval.md)
- [Stage 08 — Multi-Agent Orchestration](docs/stage-08-multi-agent.md)
- [Stage 09 — Human-in-the-Loop Systems](docs/stage-09-human-in-the-loop.md)
- [Stage 14 — Open Source + Portfolio](docs/stage-14-portfolio.md)

Beginner rules:

- Start with one model.
- Start with one tool.
- Start with one agent.
- Add complexity only after a stable baseline.

## Agent basics — what, why, and how

**What is an agent?** An AI agent is a program where an LLM decides what to do next, instead of a developer hard-coding every step. A normal program follows a fixed path you wrote in advance (`if X then Y`). An agent is given a goal and a set of tools, and *the model itself* chooses which tool to call, in what order, based on what it sees — including deciding it's done and returning an answer.

**Why use one instead of a normal script?** Reach for an agent when the steps needed to solve a task can't be known in advance — the right sequence of actions depends on what earlier steps returned. If you can write the exact steps ahead of time, write a normal script instead; it'll be cheaper, faster, and easier to debug. Agents earn their cost specifically on tasks with unpredictable branching (a question might need 0, 1, or 5 tool calls — you don't know until you're in it).

**How does one actually work?** Every agent, no matter the framework, runs the same loop:

1. **Perceive** — the model reads the current conversation/state (the user's task, plus any tool results so far).
2. **Decide** — the model reasons about what to do next: answer now, or call a tool.
3. **Act** — if it chose a tool, your code runs that tool (a calculator, a web search, a database query...) and captures the result.
4. **Observe & repeat** — the tool's result goes back to the model, and the loop returns to step 1 — until the model has enough information to give a final answer, or a safety limit (max iterations) is hit so it can't loop forever.

This is called the **ReAct pattern** (Reason + Act), and it's the same shape whether you write it by hand or a framework writes it for you.

**Minimal example — the actual decision point**, from [`examples/01-basic-agent/agent.py`](examples/01-basic-agent/agent.py):

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

## Start here: a basic AI agent

Before touching a framework, build one agent from raw API calls so the ReAct loop isn't a mystery a library is hiding from you. [`examples/01-basic-agent/`](examples/01-basic-agent/) is a ~120-line Python script: one tool (calculator), one loop, no framework. Run it, then read Stage 7 and Stage 8 knowing exactly what "framework magic" is standing in for.

Once that's comfortable, three follow-ups build on it in different directions:

- [`examples/01-basic-agent-langchain/`](examples/01-basic-agent-langchain/) — the same agent using LangChain's `ChatAnthropic` + `@tool`, but still a hand-written loop — makes concrete why LangChain alone doesn't replace the loop.
- [`examples/01-basic-agent-langgraph/`](examples/01-basic-agent-langgraph/) — the same agent, rebuilt on LangGraph's `StateGraph` instead of the raw loop, with a line-by-line comparison of what the framework replaces.
- [`examples/03-mcp-tool-server/`](examples/03-mcp-tool-server/) — the same agent again, with its one tool moved behind an MCP server, showing what MCP actually changes (and doesn't) versus the plain tool array in example 01.

Read those three in order and you get the full picture: raw API → LangChain's components (still your own loop) → LangGraph's graph (the loop becomes edges).

Once the calculator toy feels solid, two more examples do real agentic work — same task (give it a company name, get a dated, grouped news summary), two different search tools, chosen to make one distinction concrete:

- [`examples/01_web_agent_langgraph/`](examples/01_web_agent_langgraph/) — search via Claude's built-in `web_search`, a **server-side** tool: Anthropic's infrastructure runs the search, so the graph is just one node — no `ToolNode`, because there's nothing left for your code to execute.
- [`examples/01_web_agent_langgraph_tavily/`](examples/01_web_agent_langgraph_tavily/) — the identical agent and prompt, using [Tavily](https://tavily.com/) instead — a **client-side** tool, so `ToolNode` and the agent↔tools loop are back, same shape as the calculator LangGraph example.

Both refuse to fabricate a summary when search comes up empty — read them back to back and the lesson is: whether a graph needs a `ToolNode` depends entirely on whether the tool runs on your infrastructure or the provider's, not on what the tool does.

The natural next step after that toward *agentic* AI is a two-agent handoff — see the note at the bottom of example 01's README for where that lives once you build it.

## Stages

| # | Stage | Core topics |
|---|-------|--------------|
| 01 | [Python + Async Foundations](docs/stage-01-python-async.md) | asyncio, FastAPI, event-driven architecture, error handling, API integration |
| 02 | [LLM Fundamentals for Agents](docs/stage-02-llm-fundamentals.md) | Context engineering, model routing, token economics, cost optimization, latency tradeoffs, failure modes |
| 03 | [Tool Calling + Structured Outputs](docs/stage-03-tool-calling.md) | Pydantic validation, function schemas, error recovery, dynamic tool discovery, MCP vs. plain APIs, computer-use tools |
| 04 | [Memory + State Management](docs/stage-04-memory-state.md) | Short-term, persistent, and long-term memory as three distinct composable pieces |
| 05 | [RAG + Retrieval](docs/stage-05-rag-retrieval.md) | Vector databases, chunking, naive vs. hybrid vs. agentic RAG, corrective RAG (CRAG), retrieval vs. tool calls |
| 06 | [Sessions, State + Events](docs/stage-06-sessions-state.md) | The session object, state vs. events, an in-memory session service, updating context, inter-agent communication via shared sessions |
| 07 | [Single-Agent Workflows](docs/stage-07-single-agent.md) | The six Anthropic workflow patterns, Tree-of-Thought, ReAct loops, self-reflection, iteration limits, graceful degradation, long-running agents, cognitive architecture (confidence gating, attention, knowledge boundaries), computer-use/voice agents |
| 08 | [Multi-Agent Orchestration](docs/stage-08-multi-agent.md) | Framework landscape (LangGraph/CrewAI/ADK/OpenAI Agents SDK/Pydantic AI/Mastra), supervisor patterns, message passing, conflict resolution, handoffs + inline guardrails, A2A protocol |
| 09 | [Human-in-the-Loop Systems](docs/stage-09-human-in-the-loop.md) | Uncertainty detection, approval gates, audit trails, resume logic, intervention points |
| 10 | [Evaluation + Quality Assurance](docs/stage-10-evaluation-qa.md) | Automated eval harnesses, LLM-as-judge, regression testing, hallucination metrics, eval vs. red-teaming |
| 11 | [Observability + Tracing](docs/stage-11-observability.md) | Distributed tracing, cost dashboards, latency monitoring, alerting |
| 12 | [Security + Guardrails](docs/stage-12-security-guardrails.md) | Prompt injection defense, agentic-specific risks (goal hijacking, memory poisoning, cascading failures), red-teaming, output filtering, PII redaction, sandboxed execution, compliance |
| 13 | [Production Deployment](docs/stage-13-deployment.md) | vLLM/SGLang, Kubernetes scaling, CI/CD for agents, canary releases, rollback strategies |
| 14 | [Open Source + Portfolio](docs/stage-14-portfolio.md) | Ship autonomous agents publicly, architecture docs, demos, OSS contributions |

## Notes on sequencing

This path assumes working Python fluency already. If not, insert a Stage 0 covering core syntax, type hints, and virtual environments before Stage 1 — everything after depends on being comfortable reading async stack traces.

Frameworks named here (LangGraph, CrewAI, LangSmith, Arize) are one reasonable default stack, not the only correct one — the underlying patterns (supervisor orchestration, tiered memory, eval harnesses, tracing) transfer to any framework or a from-scratch implementation.

## License

Use freely — this is a personal study guide, not licensed software.
