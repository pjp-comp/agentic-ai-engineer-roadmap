# Becoming an Agentic AI Engineer

A sequenced, 13-stage curriculum from async Python fundamentals to shipping observable, guarded, production multi-agent systems. Each stage lists what to build, a runnable code brief, a completion check, and vetted sources — official docs first, courses and papers second.

- **Prepared:** 2026-07-17
- **Pace:** ~4–7 months at 8–10 hrs/week
- **Track:** Python-first
- **Live version:** [Artifact page](https://claude.ai/code/artifact/b9e34e22-9c4a-4c3f-9994-ba13f7a7b463)

## How to use this

Stages are ordered by dependency, not difficulty — Stage 7 assumes Stages 3, 5, and 6 are solid, since orchestration is just single-agent loops sharing sessions. Don't skip the "brief" in each stage: it's the smallest program that forces you to touch the real API surface, not a toy print statement.

Treat official docs as ground truth and everything else (courses, blog tutorials) as commentary on it — frameworks like LangGraph ship breaking changes often enough that an older tutorial can mislead you on current APIs.

## AI Agent vs. Agentic AI — the distinction that matters

These terms get used interchangeably, but they describe different points on a spectrum of autonomy. Knowing which one you're building changes your architecture, your error-handling budget, and how much you can trust the system unattended.

| | **AI Agent** | **Agentic AI** |
|---|---|---|
| **Scope** | One model, one task, a bounded set of tools | Multiple agents/roles, coordinated toward a broader goal |
| **Loop** | Perceive → decide → act, usually a single pass or a short bounded loop | Continuous plan → act → reflect → replan, often spanning many sub-tasks |
| **Autonomy** | Executes a defined task; a human or a fixed pipeline decides what task to run next | Decomposes the *goal itself* — decides what tasks need to exist |
| **State** | Mostly stateless per call, maybe a short conversation buffer | Persistent memory, shared state across agents and sessions |
| **Example** | A support bot that looks up an order and answers one question | A system that plans a trip, books flights, adjusts for a cancellation, and re-plans the itinerary without being told each step |
| **Where it lives in this roadmap** | Stages 1–5 build the parts; **Stage 6** is where a single agent becomes real | **Stage 7** is where agentic AI starts — orchestration, handoffs, and shared goals across agents |

The short version: **an AI agent is a component; agentic AI is a system of components pursuing a goal with minimal supervision.** A single ReAct loop that calls a weather API is an agent. A crew of agents that researches a market, drafts a strategy, and revises it based on a critic agent's feedback is agentic AI. Stage 6 in this roadmap teaches you to build the former well; Stage 7 teaches you to compose several of them into the latter.

## Start here: a basic AI agent

Before touching a framework, build one agent from raw API calls so the ReAct loop isn't a mystery a library is hiding from you. [`examples/01-basic-agent/`](examples/01-basic-agent/) is a ~120-line Python script: one tool (calculator), one loop, no framework. Run it, then read Stage 6 and Stage 7 knowing exactly what "framework magic" is standing in for.

Once that's comfortable, the natural next step toward *agentic* AI is a two-agent handoff — see the note at the bottom of the example's README for where that lives once you build it.

## Stages

| # | Stage | Core topics |
|---|-------|--------------|
| 01 | [Python + Async Foundations](docs/stage-01-python-async.md) | asyncio, FastAPI, event-driven architecture, error handling, API integration |
| 02 | [LLM Fundamentals for Agents](docs/stage-02-llm-fundamentals.md) | Context management, model routing, token economics, latency tradeoffs, failure modes |
| 03 | [Tool Calling + Structured Outputs](docs/stage-03-tool-calling.md) | Pydantic validation, function schemas, error recovery, dynamic tool discovery, MCP vs. plain APIs |
| 04 | [Memory + State Management](docs/stage-04-memory-state.md) | Short-term, persistent, and long-term memory as three distinct composable pieces |
| 05 | [Sessions, State + Events](docs/stage-05-sessions-state.md) | The session object, state vs. events, an in-memory session service, updating context, inter-agent communication via shared sessions |
| 06 | [Single-Agent Workflows](docs/stage-06-single-agent.md) | ReAct loops, plan-and-execute, self-reflection, iteration limits, graceful degradation, long-running agents |
| 07 | [Multi-Agent Orchestration](docs/stage-07-multi-agent.md) | LangGraph/CrewAI, supervisor patterns, message passing, conflict resolution, handoffs |
| 08 | [Human-in-the-Loop Systems](docs/stage-08-human-in-the-loop.md) | Uncertainty detection, approval gates, audit trails, resume logic, intervention points |
| 09 | [Evaluation + Quality Assurance](docs/stage-09-evaluation-qa.md) | Automated eval harnesses, LLM-as-judge, regression testing, hallucination metrics |
| 10 | [Observability + Tracing](docs/stage-10-observability.md) | Distributed tracing, cost dashboards, latency monitoring, alerting |
| 11 | [Security + Guardrails](docs/stage-11-security-guardrails.md) | Prompt injection defense, output filtering, PII redaction, sandboxed execution, compliance |
| 12 | [Production Deployment](docs/stage-12-deployment.md) | vLLM/SGLang, Kubernetes scaling, CI/CD for agents, canary releases, rollback strategies |
| 13 | [Open Source + Portfolio](docs/stage-13-portfolio.md) | Ship autonomous agents publicly, architecture docs, demos, OSS contributions |

## Notes on sequencing

This path assumes working Python fluency already. If not, insert a Stage 0 covering core syntax, type hints, and virtual environments before Stage 1 — everything after depends on being comfortable reading async stack traces.

Frameworks named here (LangGraph, CrewAI, LangSmith, Arize) are one reasonable default stack, not the only correct one — the underlying patterns (supervisor orchestration, tiered memory, eval harnesses, tracing) transfer to any framework or a from-scratch implementation.

## License

Use freely — this is a personal study guide, not licensed software.
