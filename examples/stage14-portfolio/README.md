[← Back to roadmap](../../README.md)

# Stage 14 — Shipping What You Already Built

[Stage 14](../../docs/stage-14-portfolio.md) isn't a new pattern to build — it's a checklist for taking the agent(s) from Stages 5–13 and making them visible to someone who isn't you. There's no `agent.py` here on purpose: the "code" for this stage already exists, scattered across every other `examples/stage*/` directory in this repo. This README is the checklist and templates Stage 14's brief asks for.

## The brief, as a checklist

- [ ] **Pick one agent** — not all of them. Stage 14's brief says "the agent built across Stages 5–13" (singular). [`stage08-supervisor-langgraph`](../stage08-supervisor-langgraph/) or [`stage09-human-in-the-loop`](../stage09-human-in-the-loop/) are the most demo-able candidates in this repo — multi-step, visibly doing something a single prompt-response can't.
- [ ] **Public repo** — if this whole repo isn't already public, that's step one. A private portfolio isn't a portfolio.
- [ ] **README with an architecture diagram** — see the template below. Write this *before* recording the demo (Stage 14's own advice) — it forces you to justify design choices made on autopilot.
- [ ] **3–5 minute demo recording** — screen recording, not slides. Show it working, then show it failing.
- [ ] **A deliberately induced failure, and how it degrades** — don't hide the failure case, stage it. A rate-limit error, a bad tool call, a timeout — then show the graceful-degradation path (Stage 7), the human-in-the-loop rejection (Stage 9), or the eval suite catching it (Stage 10) actually working.
- [ ] **One real OSS contribution** — LangGraph, CrewAI, an MCP server, anything used along the way. A docs fix counts as a start, per the brief.

## Architecture doc template

Copy this into the README of whichever example you're shipping:

```markdown
## Architecture

**What it does:** <one sentence — what does a user actually get out of this>

**Why an agent, not a script:** <per the README's "Agent basics" section —
what's unpredictable enough here that a fixed pipeline wouldn't work?>

### Diagram

<a real diagram — even ASCII is fine, see stage05/stage08's READMEs for
the START -> node -> node -> END style used throughout this repo>

### Design decisions worth explaining

- <e.g. "why a supervisor graph instead of a fixed chain" —
  see stage08's README for the actual reasoning to crib from>
- <e.g. "why this checkpointer, not that one" — see stage06/stage09>
- <e.g. "why this model, not a bigger one" — see stage02-sampling-params>

### Known limitations

<the failure mode you're about to demo on purpose belongs here, named
plainly, not discovered by the reader the hard way>
```

## Demo recording checklist

- [ ] Show the golden path first — the agent doing its job successfully, narrated in plain language (what's happening at each step, not just "and now it's thinking").
- [ ] Show the induced failure — pick one that's real and already covered somewhere in this repo, don't invent a fake one:
  - A tool call that fails (see [`stage01-async-fanout`](../stage01-async-fanout/)'s timeout/circuit-breaker, or [`stage07-react-loop`](../stage07-react-loop/)'s unreliable `lookup` tool)
  - A prompt-injection attempt in tool output (see [`stage12-guardrails`](../stage12-guardrails/))
  - A high-risk action correctly paused for human approval (see [`stage09-human-in-the-loop`](../stage09-human-in-the-loop/))
- [ ] Show the recovery — graceful degradation, a retry, a rejection, whatever the actual mechanism is. Point at the code that makes it happen, briefly.
- [ ] Keep it 3–5 minutes. Cut dead air; a demo that drags loses the point it's trying to make.

## Where the actual proof lives

Every checkbox above maps to something already built and verified in this repo:

| Stage 14 asks for | Already built at |
|---|---|
| A multi-step agent worth demoing | [`stage08-supervisor-langgraph`](../stage08-supervisor-langgraph/), [`stage09-human-in-the-loop`](../stage09-human-in-the-loop/) |
| A visible, honest failure mode | [`stage07-react-loop`](../stage07-react-loop/)'s graceful degradation, [`stage12-guardrails`](../stage12-guardrails/)'s injection log |
| Evidence it's tested, not just demoed once | [`stage10-eval-harness`](../stage10-eval-harness/) |
| Evidence it's observable | [`stage11-observability`](../stage11-observability/) |
| Evidence it's production-minded | [`stage13-idempotent-tools`](../stage13-idempotent-tools/) |

Ship one of these — polished, documented, with a recorded failure — rather than starting a fourteenth thing from scratch.
