[← Stage 13](stage-13-deployment.md) · Stage 14 of 14

# Stage 14 — Open Source + Portfolio

Ship autonomous agents publicly · write architecture docs · record demos · contribute to libs

## Why this matters

Everything in Stages 1–13 is proof of skill only once someone else can see it working. A portfolio piece that's live, documented, and has a recorded failure mode you fixed is worth more than a private repo — it shows judgment under real constraints, not just code that runs once.

## Brief

Take the agent built across Stages 5–13 and ship it: public repo, a README with an architecture diagram, a 3–5 minute demo recording including a deliberately induced failure (rate limit, bad tool call) and how it degrades. Then make one real contribution to a library you used (LangGraph, CrewAI, an MCP server) — a docs fix counts as a start.

**Checklist + templates:** [`examples/stage14-portfolio/`](../examples/stage14-portfolio/) turns this brief into an actual checklist, an architecture-doc template, and a demo-recording checklist — plus a table mapping each requirement to an already-built, already-verified example elsewhere in this repo (a failure mode to demo, evidence of testing, evidence of observability) so shipping doesn't mean starting a fourteenth project from scratch.

## Sources

| Type | Resource |
|------|----------|
| Repo | [langchain-ai/langgraph — good first issues](https://github.com/langchain-ai/langgraph) |
| Repo | [crewAIInc/crewAI — contribution guide](https://github.com/crewAIInc/crewAI) |
| Course | [Hugging Face — AI Agents Course (free, includes certification)](https://huggingface.co/learn/agents-course) |
| Practice | Write the architecture doc before the demo — it forces you to justify design choices you may have made on autopilot |

## Done when

A stranger can read your README, understand what the agent does and why it's built that way, and run it themselves.

---
[← Stage 13 — Production Deployment](stage-13-deployment.md) · [Back to roadmap](../README.md)
