# Becoming an Agentic AI Engineer

A sequenced, 12-stage curriculum from async Python fundamentals to shipping observable, guarded, production multi-agent systems. Each stage lists what to build, a runnable code brief, a completion check, and vetted sources — official docs first, courses and papers second.

- **Prepared:** 2026-07-17
- **Pace:** ~4–7 months at 8–10 hrs/week
- **Track:** Python-first
- **Live version:** [Artifact page](https://claude.ai/code/artifact/b9e34e22-9c4a-4c3f-9994-ba13f7a7b463)

## How to use this

Stages are ordered by dependency, not difficulty — Stage 6 assumes Stage 3 and 5 are solid, since orchestration is just single-agent loops wired together. Don't skip the "brief" in each stage: it's the smallest program that forces you to touch the real API surface, not a toy print statement.

Treat official docs as ground truth and everything else (courses, blog tutorials) as commentary on it — frameworks like LangGraph ship breaking changes often enough that an older tutorial can mislead you on current APIs.

## Stages

| # | Stage | Core topics |
|---|-------|--------------|
| 01 | [Python + Async Foundations](docs/stage-01-python-async.md) | asyncio, FastAPI, event-driven architecture, error handling, API integration |
| 02 | [LLM Fundamentals for Agents](docs/stage-02-llm-fundamentals.md) | Context management, model routing, token economics, latency tradeoffs, failure modes |
| 03 | [Tool Calling + Structured Outputs](docs/stage-03-tool-calling.md) | Pydantic validation, function schemas, error recovery, dynamic tool discovery |
| 04 | [Memory + State Management](docs/stage-04-memory-state.md) | Short-term buffers, vector recall, context compression, cross-session sync |
| 05 | [Single-Agent Workflows](docs/stage-05-single-agent.md) | ReAct loops, plan-and-execute, self-reflection, iteration limits, graceful degradation |
| 06 | [Multi-Agent Orchestration](docs/stage-06-multi-agent.md) | LangGraph/CrewAI, supervisor patterns, message passing, conflict resolution, handoffs |
| 07 | [Human-in-the-Loop Systems](docs/stage-07-human-in-the-loop.md) | Uncertainty detection, approval gates, audit trails, resume logic, intervention points |
| 08 | [Evaluation + Quality Assurance](docs/stage-08-evaluation-qa.md) | Automated eval harnesses, LLM-as-judge, regression testing, hallucination metrics |
| 09 | [Observability + Tracing](docs/stage-09-observability.md) | Distributed tracing, cost dashboards, latency monitoring, alerting |
| 10 | [Security + Guardrails](docs/stage-10-security-guardrails.md) | Prompt injection defense, output filtering, PII redaction, sandboxed execution, compliance |
| 11 | [Production Deployment](docs/stage-11-deployment.md) | vLLM/SGLang, Kubernetes scaling, CI/CD for agents, canary releases, rollback strategies |
| 12 | [Open Source + Portfolio](docs/stage-12-portfolio.md) | Ship autonomous agents publicly, architecture docs, demos, OSS contributions |

## Notes on sequencing

This path assumes working Python fluency already. If not, insert a Stage 0 covering core syntax, type hints, and virtual environments before Stage 1 — everything after depends on being comfortable reading async stack traces.

Frameworks named here (LangGraph, CrewAI, LangSmith, Arize) are one reasonable default stack, not the only correct one — the underlying patterns (supervisor orchestration, tiered memory, eval harnesses, tracing) transfer to any framework or a from-scratch implementation.

## License

Use freely — this is a personal study guide, not licensed software.
