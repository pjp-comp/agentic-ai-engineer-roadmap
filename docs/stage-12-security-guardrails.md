[← Stage 11](stage-11-observability.md) · Stage 12 of 14 · [Next: Stage 13 →](stage-13-deployment.md)

# Stage 12 — Security and Guardrails

## First pass

**Before this lesson:** Stage 03 tool dispatch; read this before giving any tool real side effects.

**Do now:** Run the guardrails example and distinguish deterministic permission blocks from model-dependent injection resistance.

**Try it yourself:** Try a forbidden role/tool pair, an unknown role, and an expression containing a function call. All must be rejected by code.

**Check your understanding:** Show that unauthorized tools never execute. Explain why pattern matching and content labels do not prove injection safety.

**Later:** Adversarial campaigns, memory poisoning, and production execution isolation. The calculator accepts a bounded arithmetic grammar; it is not a general code sandbox.

**Next in the beginner path:** [Stage 09](stage-09-human-in-the-loop.md).

Topic numbers stay stable for reference. Follow the [beginner learning order](../README.md#the-learning-flow); the previous/next links follow numerical topic order. Code in this guide is a design sketch unless labeled runnable.



## Why this matters

A tool result, retrieved document, or stored fact can contain instructions written by someone other than the user. Treat that material as untrusted data. Assume a model can follow a malicious instruction despite a well-written prompt, and enforce permissions in application code.

## Brief

Build three separate controls:

1. Label untrusted tool output and log suspicious patterns. Pattern matching misses paraphrases and can flag harmless quotations; it is a diagnostic signal.
2. Check an authenticated caller's allowed tool names and argument scope **before execution**. A role supplied by the model is not an authenticated identity.
3. For a calculator, accept a small arithmetic grammar rather than executing Python. General code execution requires a separately designed isolation boundary.

**Runnable version:** [stage12-guardrails](../examples/stage12-guardrails/) demonstrates labels, a hard-coded role allowlist, and a bounded arithmetic interpreter. The roles and order backend are mocks. It does not implement user authentication, PII redaction, or a general code sandbox.

## What labeling does and does not prove

The demo retains tool text inside an untrusted-content label. This communicates intent to the model but does not remove malicious instructions or prevent a model from obeying them. A successful answer on the included attack is one observation, not proof of injection prevention. Keep secrets out of prompts, minimize tool privileges, validate results, and require approval for consequential actions.

## Beyond prompt injection — risks specific to agentic systems

- **Goal hijacking:** untrusted content redirects the task.
- **Memory poisoning:** a malicious or incorrect fact persists into later sessions.
- **Unsafe handoffs:** one agent treats another agent's output as privileged instructions.
- **Cascading failures:** an unchecked mistake becomes input to several later actions.

Validate source provenance and permissions at every handoff. Give stored facts correction and deletion paths. Use [Stage 04](stage-04-memory-state.md) for memory lifetimes and [Stage 09](stage-09-human-in-the-loop.md) for approval gates.

## Testing for these — red-teaming, not just eval

Run both legitimate requests and adversarial variations. Assert deterministic properties first: a forbidden tool never executes, unknown roles have no permissions, and invalid arithmetic syntax is rejected. Then inspect model behavior under injected tool text and poisoned memories. Keep successful attacks as regression cases in [Stage 10](stage-10-evaluation-qa.md), document remaining failures, and rerun after changes. Passing a finite test set is not a universal safety guarantee.

## Execution boundaries

An ordinary subprocess inherits host capabilities; removing `__builtins__` from `eval` does not turn Python into a secure language sandbox. The demo avoids that approach entirely. Its calculator interprets numeric literals, parentheses, and `+ - * / // %`, with input, tree-size, and magnitude limits. It rejects calls, attributes, imports, and exponentiation. This narrows the accepted language; it does not isolate arbitrary code.

For actual generated-code execution, define filesystem, network, credential, CPU, memory, and process limits and choose an isolation technology appropriate to that threat model. See [Stage 13](stage-13-deployment.md#agent-native-infrastructure--what-agents-need-that-ordinary-services-dont).

## Sources

- [Python built-in functions: eval](https://docs.python.org/3/library/functions.html#eval) — arbitrary code execution and untrusted-input risks.
- [OWASP LLM application risks](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — threat categories and mitigation guidance.
- [Anthropic tool-use documentation](https://platform.claude.com/docs/en/build-with-claude/tool-use) — provider-specific tool contracts; consult the version matching your dependencies.

## Done when

Show a denied tool call blocked before execution, an invalid expression rejected by the interpreter, and an injection attempt recorded and assessed. Explain which controls are enforced by code and which outcomes still depend on the model. Record unresolved risks rather than declaring the system injection-proof.

---
[Back to learning path](../README.md) · [Next: Stage 13 →](stage-13-deployment.md)
