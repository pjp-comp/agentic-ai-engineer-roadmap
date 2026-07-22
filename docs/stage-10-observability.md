[← Stage 09](stage-09-evaluation-qa.md) · Stage 10 of 13 · **Next:** [Stage 11 →](stage-11-security-guardrails.md)

# Stage 10 — Observability + Tracing

Distributed tracing (LangSmith/Arize) · cost dashboards · latency monitoring · alerting

## Why this matters

A multi-agent run spans dozens of LLM calls, tool invocations, and state transitions — debugging it from logs alone doesn't scale. Tracing gives you the full call tree: which agent, which prompt, which tokens, which cost, per step.

## Brief

Instrument the Stage 6 supervisor graph with tracing so every node emits a span with input/output, token counts, latency, and cost; build one dashboard view that surfaces p95 latency and $/run.

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agent-stage9"
# every graph.invoke() now auto-traces to LangSmith;
# or use OpenTelemetry exporters for a vendor-neutral pipeline
```

## Sources

| Type | Resource |
|------|----------|
| Docs | [LangSmith — Observability docs](https://docs.langchain.com/langsmith/observability) |
| Guide | [Agent tracing with OpenTelemetry + LangSmith](https://callsphere.ai/blog/ai-agent-observability-opentelemetry-langsmith-tracing) |
| Comparison | [LangSmith vs. Arize — evals & deployment compared](https://www.langchain.com/resources/langsmith-vs-arize) |
| Tool | Langfuse (open-source alternative), Arize Phoenix (ML-rigor drift detection) |

## Done when

Given a bad output, you can find the exact span (prompt, tool call, or handoff) that caused it in under two minutes.

---
[← Stage 09 — Evaluation + Quality Assurance](stage-09-evaluation-qa.md) · [Back to roadmap](../README.md) · **Next:** [Stage 11 — Security + Guardrails →](stage-11-security-guardrails.md)
