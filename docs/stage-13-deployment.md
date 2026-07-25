[← Stage 12](stage-12-security-guardrails.md) · Stage 13 of 14 · **Next:** [Stage 14 →](stage-14-portfolio.md)

# Stage 13 — Production Deployment

vLLM/SGLang · Kubernetes scaling · CI/CD for agents · canary releases · rollback strategies

## Why this matters

Kubernetes earns its keep only once you need horizontal scaling, multi-tenant isolation, or rolling deploys — not by default. If self-hosting open-weight models, vLLM (broad compatibility, Apache-2.0) and SGLang (higher throughput, structured-generation-first) are the two serving engines worth knowing. Version model weights, serving config, and prompt templates independently — all three affect output quality and drift separately.

## Brief

Containerize the Stage 9 agent service; write a CI pipeline that runs the Stage 10 eval suite as a merge gate; deploy behind a canary (5% traffic) with automatic rollback if error rate or eval pass-rate regresses.

```yaml
# CI gate (excerpt)
- run: pytest tests/eval_suite.py --min-pass-rate=0.92
- run: docker build -t agent:$SHA .
- run: kubectl set image deployment/agent-canary agent=agent:$SHA
- run: ./scripts/watch_canary.sh --rollback-on-error-rate=0.02
```

## Sources

| Type | Resource |
|------|----------|
| Docs | [vLLM — Using Kubernetes (official docs)](https://docs.vllm.ai/en/stable/deployment/k8s/) |
| Guide | [vLLM Production Deployment — complete 2026 guide](https://www.sitepoint.com/vllm-production-deployment-guide-2026/) |
| Guide | [SGLang — the complete guide to high-performance LLM inference](https://inference.net/content/sglang-complete-guide/) |
| Reference | LiteLLM as an OpenAI-compatible gateway in front of multiple backends |

## Done when

A bad deploy auto-rolls-back before it reaches full traffic, with no manual intervention.

---
[← Stage 12 — Security + Guardrails](stage-12-security-guardrails.md) · [Back to roadmap](../README.md) · **Next:** [Stage 14 — Open Source + Portfolio →](stage-14-portfolio.md)
