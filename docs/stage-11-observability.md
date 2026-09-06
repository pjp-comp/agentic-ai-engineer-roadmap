[← Stage 10](stage-10-evaluation-qa.md) · Stage 11 of 14 · **Next:** [Stage 12 →](stage-12-security-guardrails.md)

# Stage 11 — Observability + Tracing

Distributed tracing (LangSmith/Arize) · spans, traces, and span attributes · what to instrument · cost dashboards · latency monitoring · alerting on agent-specific signals

## Why this matters

A multi-agent run spans dozens of LLM calls, tool invocations, and state transitions — debugging it from logs alone doesn't scale. Tracing gives you the full call tree: which agent, which prompt, which tokens, which cost, per step.

The deeper reason this stage exists: **an agent has no stack trace.** When ordinary software misbehaves, the exception tells you where. When an agent misbehaves, there's no error at all — it ran happily to completion and produced something wrong. The trace *is* the stack trace you don't otherwise get, reconstructed after the fact. Without it, debugging degenerates into re-running the agent and hoping it misbehaves the same way twice, which at `temperature > 0` it often won't.

## The vocabulary: traces, spans, and attributes

Three terms carry all of it, and they're borrowed intact from ordinary distributed tracing (OpenTelemetry) rather than invented for agents:

| Term | What it is | In an agent |
|---|---|---|
| **Trace** | One complete end-to-end operation, identified by a `trace_id` | One agent run, from user request to final answer |
| **Span** | One unit of work inside a trace, with a start time, end time, and a parent span | One LLM call, one tool execution, one graph node |
| **Attributes** | Key/value metadata attached to a span | `model`, `tokens_in`, `tokens_out`, `cost_usd`, `tool_name`, `node_name` |

Spans nest into a tree via parent pointers, and that tree is the whole value proposition: a flat log tells you *what happened*, a span tree tells you *what happened inside what*. When a supervisor's writer node produces nonsense, the span tree shows you the researcher span that fed it — a flat log just shows two events near each other in time and leaves you to guess at the causality.

**The one rule that makes traces useful later:** a `trace_id` must be generated once at the entry point and propagated through every call in the run. Generating one per LLM call gives you a hundred single-span traces and no tree at all — which is the most common way a first tracing implementation ends up useless.

## What to instrument — and what each signal actually catches

Not everything deserves a span, and the useful attributes aren't obvious until you've had to debug without them:

| Span | Attributes worth recording | The failure it catches |
|---|---|---|
| **LLM call** | model, tokens_in/out, cached tokens, latency, cost, stop reason, temperature | Cost spikes; silent truncation (`stop_reason == "max_tokens"` is *not* an error — the request succeeds and the answer is quietly cut off) |
| **Tool call** | tool name, arguments, result, error flag, duration | The single most valuable span type. Redundant calls, wrong tool selection, and tools that fail silently and return a plausible-looking error string |
| **Graph node / agent step** | node name, iteration number, state size | Loops that stagnate; state growing unboundedly across steps |
| **Retrieval** | query, chunks returned, scores | Answers grounded in bad context — indistinguishable from a reasoning failure without this |
| **Guardrail / validation** | rule triggered, action taken | Guardrails silently firing (or silently not firing) in production |

Two attributes that matter more than they look: **cached-token counts**, because a cache-hit-rate collapse is invisible in latency but doubles your bill ([Stage 2](stage-02-llm-fundamentals.md)); and **iteration number** on every span, because "the agent did the right thing on attempt 4" and "the agent did the right thing immediately" look identical in a final output but are very different systems.

**Record inputs and outputs, not just metrics.** A span saying "the LLM call took 1.4s and cost $0.003" cannot tell you why the answer was wrong. A span containing the actual prompt and the actual response can. This is what separates agent tracing from ordinary APM — and it's also why the PII warning below is not optional.

## Brief

Instrument the Stage 8 supervisor graph with tracing so every node emits a span with input/output, token counts, latency, and cost; build one dashboard view that surfaces p95 latency and $/run.

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agent-stage9"
# every graph.invoke() now auto-traces to LangSmith;
# or use OpenTelemetry exporters for a vendor-neutral pipeline
```

**Runnable version:** [`examples/stage11-observability/`](../examples/stage11-observability/) instruments the exact Stage 8 supervisor graph this brief describes with real spans (input/output, real token counts from Ollama, latency, cost) written to a local trace log — no hosted backend required — plus a dashboard script for p95 latency/$-per-run and a `--trace <id>` lookup for finding the exact span behind a bad output. Runs entirely on Ollama.

## OpenTelemetry vs. a vendor SDK — the one architectural decision here

The brief's two-line `LANGCHAIN_TRACING_V2` snippet is the fastest path to working traces, and for learning it's the right one. The decision it hides is worth making deliberately once you're past that:

| | **Vendor SDK** (LangSmith, Langfuse, Arize) | **OpenTelemetry** |
|---|---|---|
| **Setup** | Minutes — often one env var | Longer: exporter, collector, a backend to send to |
| **Agent-specific UI** | Built for this — prompt diffing, per-run replay, eval integration | Generic span viewer unless the backend adds LLM-aware views |
| **Lock-in** | Your instrumentation is vendor-shaped | Instrument once, swap backends by changing an exporter |
| **Correlates with the rest of your stack** | Usually not — agent traces live in one tool, service traces in another | Yes: the agent's spans join the same trace as the API request and DB query that triggered it |

The convergence worth knowing: **OpenTelemetry now has GenAI semantic conventions** — a standard set of span/attribute names (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, and so on) — and the major vendors accept OTel data. So this is decreasingly either/or. The pragmatic path: use a vendor SDK while learning, adopt OTel semantic conventions for your *attribute names* even if you're exporting to a vendor, and you keep the option to switch later at nearly no cost.

**Where the split genuinely matters:** if your agent is one component inside a larger service, OTel is the right answer, because the ability to see a slow API request and the agent run that caused it *in one trace* is the entire point of distributed tracing. If the agent is the whole product, a vendor SDK's purpose-built UI usually wins.

## Alerting on agent-specific signals

Standard service alerts (error rate, p95 latency, availability) still apply and still matter. But an agent's characteristic failure is **not an error** — it's a successful response that's wrong, expensive, or degenerate. Those need their own signals:

- **Cost per run, p95 not mean.** A mean hides the run that looped 40 times. Alert on the tail.
- **Iterations per run.** A rising average means the agent is increasingly failing to converge on the first try — the leading indicator of a prompt or tool regression, visible before the pass rate drops.
- **Tool error rate, per tool.** A single tool degrading is invisible in an overall success rate, because the agent often routes around it and still answers — more expensively and less accurately.
- **Cache hit rate.** A drop is a pure cost regression with no latency or quality symptom to alert you.
- **Degradation rate.** How often the run ends in [Stage 7's graceful-degradation path](stage-07-single-agent.md#long-running-agents) rather than a real answer. This is the agent equivalent of an error rate, and it's the one most teams never instrument.
- **Guardrail trigger rate.** A sudden spike is either an attack ([Stage 12](stage-12-security-guardrails.md)) or a broken rule. Both want investigating.

**The connection to Stage 10:** these are online signals from real traffic; [Stage 10's evals](stage-10-evaluation-qa.md) are offline checks against a fixed dataset. You need both, and they catch different things — evals catch regressions before deploy, these catch the drift that only shows up against real user input. The trajectory-eval measures in Stage 10 are computed from exactly the spans described above, which is the practical argument for building this stage before that one.

## One warning: traces contain everything

Recording span inputs and outputs is what makes tracing useful, and it means **your trace store now holds every prompt and every response** — including whatever personal data users typed and whatever your tools returned from a database. That store is often a third-party SaaS.

Three things to settle before this reaches production, not after: **redact at the source** (scrub PII before the span is exported, not in the UI — [Stage 12's PII redaction](stage-12-security-guardrails.md) is the same code, applied at a different boundary); **set a retention policy** (traces are high-volume and rarely useful after a couple of weeks); and **sample** (100% tracing is affordable at low volume and ruinous at high — sample normal runs, but always keep the ones that errored or hit a degradation path, since those are the only ones you'll want to read).

## Sources

| Type | Resource |
|------|----------|
| Docs | [OpenTelemetry — GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the standard attribute names worth adopting even on a vendor backend |
| Docs | [LangSmith — Observability docs](https://docs.langchain.com/langsmith/observability) |
| Guide | [Agent tracing with OpenTelemetry + LangSmith](https://callsphere.ai/blog/ai-agent-observability-opentelemetry-langsmith-tracing) |
| Comparison | [LangSmith vs. Arize — evals & deployment compared](https://www.langchain.com/resources/langsmith-vs-arize) |
| Tool | Langfuse (open-source alternative), Arize Phoenix (ML-rigor drift detection) |

## Done when

Given a bad output, you can find the exact span (prompt, tool call, or handoff) that caused it in under two minutes.

---
[← Stage 10 — Evaluation + Quality Assurance](stage-10-evaluation-qa.md) · [Back to roadmap](../README.md) · **Next:** [Stage 12 — Security + Guardrails →](stage-12-security-guardrails.md)
