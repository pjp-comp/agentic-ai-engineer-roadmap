[← Learning path](../README.md)

# Repository teaching review

Reviewed on 2026-10-28. This is a learning repository, with independent examples rather than an integrated production agent service.

## Learning-flow decisions

The first-pass route is Python → model fundamentals → raw tool loop → bounded control → memory → sessions → permissions → human approval → tracing → evaluation → deployment concepts → portfolio. Stage numbers remain stable; they identify topics, not the recommended first-pass order.

The LangGraph calculator comparison is a prerequisite for the approval demo. The bundled observability demo depends on the multi-agent supervisor, so a beginner first applies basic tracing to a single agent and defers that demo. Retrieval, model routing, vector memory, MCP, multimodal processing, and multi-agent orchestration are extensions. Evaluation and logging begin with the first working tool loop; the later lessons deepen them.

Every stage has prerequisites, first practice, an exercise, a completion check, and deferred topics. The catalog accounts for all 25 Python projects and both documentation-only example directories.

## Consolidation and corrections

- Replaced repeated root setup, example descriptions, and competing starting points with one route, one setup walkthrough, and one example catalog.
- Kept raw SDK, LangChain, LangGraph, and provider-switch comparisons because they teach different abstractions. No example directories or saved learner data were deleted.
- Corrected the claim that agentic AI requires multiple agents, and distinguished context/token limits from loop-step limits.
- Replaced the static model-price table and guaranteed historical routing savings with measured-cost guidance. Hosted routing now requires an explicitly configured large-model ID, with configurable illustrative rates.
- Corrected memory completion criteria: a bounded window intentionally forgets old messages; storage is separate from what reaches the model.
- Replaced unsafe `eval` in the async endpoint and the purported security sandbox with bounded arithmetic interpreters. Content labels and regex patterns are described as fallible signals, not injection prevention.
- Changed the retry demo to use logical action identity, allowing distinct purchases with identical inputs. Documented the in-memory ledger's lack of crash/concurrency guarantees.
- Clarified that deployment snippets, portfolio checklists, and example outputs are plans or illustrations rather than completed production validation.

## Checks and their limits

Run repository syntax and local-link checks from the root:

```bash
python scripts/check_repository.py
```

Run regression checks from each named example directory:

```bash
# examples/stage01-async-fanout
uv run python -m unittest -v test_app.py
# examples/stage12-guardrails
uv run python -m unittest -v test_agent.py
# examples/stage13-idempotent-tools
uv run python -m unittest -v test_agent.py
```

During this review all 10 regression tests passed, using the existing per-example Python environments where dependencies were needed. The no-model retry/cache demo also ran successfully. Python syntax and Markdown local-file/heading links are checked across first-party repository files; installed virtual environments and ignored learner stores are excluded.

Live Ollama access was denied by the execution environment. Model-dependent examples, provider credentials, PDF extraction on user documents, external-link availability across the entire reading list, and real deployment behavior were not end-to-end validated. Older measurements in advanced example READMEs are historical observations, not newly reproduced evidence. Locked dependencies support reproducibility but do not establish compatibility with every newer provider or framework release.

Official references consulted for the conceptual corrections include [Anthropic's workflow/agent distinction](https://www.anthropic.com/engineering/building-effective-agents), [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling), [Python's eval documentation](https://docs.python.org/3/library/functions.html#eval), and [Stripe idempotency](https://docs.stripe.com/api/idempotent_requests).
