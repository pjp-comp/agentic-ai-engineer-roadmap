[← Learning path](../../README.md) · [Stage 13 lesson](../../docs/stage-13-deployment.md)

# Retry identity and read caching

No model or API key is required. The mock ledger keeps its state in memory; this demonstrates a lost response followed by a retry while the backend stays alive. It does not simulate an actual process crash or durable recovery.

```bash
cd examples/stage13-idempotent-tools
uv run agent.py
```

## What to observe

1. `purchase-001` charges $49.99 once. Retrying the same action returns its original result.
2. `purchase-002` charges the same customer the same amount again. It is a new purchase, so the total becomes two charges.
3. Repeated Paris weather lookup hits a TTL cache; Tokyo performs a separate lookup. Weather results are mock data.

`charge_card(amount, customer_id, action_id)` uses action identity to distinguish retries from new requests. Keep an identifier stable for one logical action. It rejects reuse with different arguments. `idempotency_key(tool_name, tool_input)` is a content fingerprint used for payload consistency and read-cache keys; content alone is not purchase identity.

## Try it yourself

Call `charge_card` again with `purchase-001` and a different amount: expect `ValueError`, with no new charge. Create a new `ChargeLedger` and retry the old identifier: its empty dictionary cannot remember the earlier action. Explain why a durable backend is needed.

The mock is not concurrency-safe and has a gap between effect execution and saving the result. For real external actions, use the downstream service's idempotency support or a durable transaction design; saving a local dictionary after an external call does not close that gap. See [Stripe's API contract](https://docs.stripe.com/api/idempotent_requests).
