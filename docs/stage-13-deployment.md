[← Stage 12](stage-12-security-guardrails.md) · Stage 13 of 14 · [Next: Stage 14 →](stage-14-portfolio.md)

# Stage 13 — Production Deployment

## First pass

**Before this lesson:** A working agent plus approval, evaluation, and tracing concepts.

**Do now:** Run the no-model idempotent-tools example. Keep one action identifier on retry and use a different one for a new purchase.

**Try it yourself:** Submit identical inputs as two distinct actions; both should occur. Reuse an action identifier with changed inputs; it should fail.

**Check your understanding:** Distinguish retry identity, read caching, and durable backend guarantees. The local ledger is in-memory.

**Later:** Container services, CI deployment gates, canaries, Kubernetes, gateways, and durable execution engines. The snippets here are infrastructure plans, not included runnable deployment files.

**Next in the beginner path:** [Stage 14](stage-14-portfolio.md).

Topic numbers stay stable for reference. Follow the [beginner learning order](../README.md#the-learning-flow); the previous/next links follow numerical topic order. Code in this guide is a design sketch unless labeled runnable.



## Start with one service

A local demo is not a deployed service. First define its request/response interface, persistent stores, credentials, timeouts, run budgets, health checks, and cancellation behavior. Package one working agent and run its evaluation checks before adding scaling infrastructure. Kubernetes and self-hosted model serving are optional responses to measured needs.

The repository has no complete deployment service, Dockerfile, canary controller, or deployment CI pipeline. The following is an implementation plan:

1. Wrap the chosen agent in a service; protect it with authentication and per-user authorization.
2. Persist sessions and action records; test restart behavior and concurrent workers.
3. Run deterministic tests and quality evaluations in CI; use the actual harness options documented in Stage 10.
4. Deploy a small release, compare error rate, latency, cost, and quality with a baseline, and exercise rollback.

## Idempotent tools + tool caching — why retries are dangerous by default

A timeout means the caller may not know whether an action happened. Retrying a payment, email, or write can repeat the effect.

**Action identity:** create an identifier once for one intended action and preserve it on retries. A random UUID is suitable if saved before execution and reused. Two distinct purchases must have different identifiers even when customer and amount are identical. Hashing only the inputs would incorrectly suppress the second purchase.

**Backend enforcement:** the service performing the action must recognize that identifier, reject changed arguments for it, and return the original result on a retry. Real guarantees require durable storage, concurrency control, and an atomic relationship between the effect and its idempotency record—or the downstream provider's own idempotency support. A caller-side dictionary cannot make an external payment exactly-once.

```python
# Design sketch: create once, persist, and reuse after a timeout.
action_id = stored_action.id
payments.charge(amount=stored_action.amount, idempotency_key=action_id)
```

**Read caching:** a read-only tool may instead cache by input content with a TTL. Cache scope must include relevant tenant identity, permissions, and data freshness. Caching saves repeated work; it is not an authorization mechanism or a side-effect guarantee.

**Runnable version:** [stage13-idempotent-tools](../examples/stage13-idempotent-tools/) demonstrates one retry, a distinct purchase with identical values, and read caching. Its ledger is in memory. It simulates a lost caller response; it does not kill a process or prove recovery after backend restart.

## Agent-native infrastructure — what agents need that ordinary services don't

| Problem | Category to investigate | What still needs design |
|---|---|---|
| A run must survive deployment or wait for a person | Durable execution / checkpointing | Replay boundaries, retry identity, cancellation, state migrations |
| Model-generated code must execute | Isolation runtime | Filesystem/network policy, credentials, resources, escape resistance |
| Multiple model providers need common controls | Gateway | Budgets, failover semantics, logging, provider-specific capabilities |

Durable execution reuses recorded progress but cannot automatically make arbitrary external side effects safe. Checkpointing and backend idempotency solve different parts of the problem.

A subprocess alone is not isolation. Containers, syscall sandboxes, microVMs, and hosted execution services have different boundaries and configurations. Choose against an explicit threat model rather than assuming any product label guarantees containment. For a calculator, a bounded grammar avoids the need to execute generated Python at all.

Gateways can centralize routing and accounting; they do not make every provider's tool, streaming, and structured-output behavior identical. Keep provider-specific evaluation coverage.

## Sources

- [Stripe: idempotent requests](https://docs.stripe.com/api/idempotent_requests) — action keys, retries, and parameter consistency.
- [Temporal: durable execution for AI](https://temporal.io/solutions/ai) — an implementation option for long-running work.
- [gVisor architecture](https://gvisor.dev/docs/architecture_guide/intro/) — one isolation design and its boundary.
- [LiteLLM gateway documentation](https://docs.litellm.ai/docs/simple_proxy) — an optional provider gateway.

## Done when

For the local exercise, a retry repeats no charge, a new action with identical inputs creates a new charge, and reusing an identifier with different inputs fails. For a deployed project, separately test backend restart, concurrent duplicate requests, uncertain external outcomes, and rollback. Report these as distinct checks.

---
[Back to learning path](../README.md) · [Next: Stage 14 →](stage-14-portfolio.md)
