[← Stage 13](stage-13-deployment.md) · Stage 14 of 14

# Stage 14 — Open Source + Portfolio

## First pass

**Before this lesson:** One finished project with recorded success and failure checks.

**Do now:** Use the portfolio checklist to document the problem, setup, architecture, limitations, and a repeatable demo.

**Try it yourself:** Ask someone to follow the setup with a fresh environment and no access to your local files or credentials.

**Check your understanding:** They can run it, observe a failure path, and understand the limits. Only claim checks you actually ran.

**Later:** Public hosting and open-source contributions; a reproducible local project is a valid first milestone.

**Next in the beginner path:** finish the portfolio checklist, then choose an extension from the [learning path](../README.md#extensions-after-the-first-agent).

Topic numbers stay stable for reference. Follow the [beginner learning order](../README.md#the-learning-flow); the previous/next links follow numerical topic order. Code in this guide is a design sketch unless labeled runnable.


Ship autonomous agents publicly · write architecture docs · record demos · contribute to libs

## Why this matters

A finished project should let another person understand your choices and reproduce your results. A documented local demo with a recorded failure path is a useful first milestone; public hosting is optional.

## Brief

Choose one working example or your own integrated project. Write setup instructions, an architecture diagram, limitations, and the checks you ran. Record a short demo including an induced failure and recovery. Optionally publish it or contribute a documentation improvement to a library you used; review credentials, data permissions, and licensing before sharing.

**Checklist + templates:** [`examples/stage14-portfolio/`](../examples/stage14-portfolio/) turns this brief into an actual checklist, an architecture-doc template, and a demo-recording checklist — plus a table mapping each requirement to an existing example elsewhere in this repo (a failure mode to demo, evidence of testing, evidence of observability) so shipping doesn't mean starting a fourteenth project from scratch.

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
