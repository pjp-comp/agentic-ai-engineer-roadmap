[← Stage 11](stage-11-observability.md) · Stage 12 of 14 · **Next:** [Stage 13 →](stage-13-deployment.md)

# Stage 12 — Security + Guardrails

Prompt injection defense · agentic-specific risks (goal hijacking, memory poisoning, cascading failures) · red-teaming · output filtering · PII redaction · sandboxed execution · compliance

## Why this matters

Prompt injection has held the #1 spot on OWASP's LLM Top 10 across every edition. Once an agent has tools with real side effects, untrusted content (a webpage, an email, a file) becomes an attack surface. Assume the model will occasionally be tricked, and design containment — least-privilege tools, output validation, sandboxing — rather than relying on the model to "just not fall for it."

## Beginner focus

- Detect and flag prompt-injection patterns in tool output.
- Use a strict tool allowlist per role.
- Block high-risk actions by default.
- Add full red-teaming after baseline guardrails work.

## Brief

Add a guard layer: strip/flag instructions found inside tool-returned content before it re-enters the prompt, enforce a tool allowlist per agent role, and run any code-execution tool in a sandboxed subprocess with no filesystem/network access by default.

```python
def sanitize_tool_output(text: str) -> str:
    if INJECTION_PATTERNS.search(text):
        log_security_event("possible_injection", text)
    return wrap_as_untrusted_data(text)  # tag so the model
    # is instructed to treat it as data, never as instructions
```

**Runnable version:** [`examples/stage12-guardrails/`](../examples/stage12-guardrails/) builds all three pieces of this brief for real — injection detection + neutralization, a per-role tool allowlist enforced in code, and a genuinely sandboxed subprocess for code execution — then runs five scenarios including the actual "ignore previous instructions" attack this stage's "Done when" names. Runs entirely on Ollama.

## Beyond prompt injection — risks specific to agentic systems

The OWASP LLM Top 10 above is written for single-model applications. Once an agent has memory, tools, and (per Stage 8) other agents to coordinate with, a separate list applies: the **OWASP Top 10 for Agentic Applications** (endorsed by NIST, Microsoft, and NVIDIA). The risks that don't reduce to "sanitize tool output" from this stage's brief:

- **Goal hijacking** — an attacker doesn't inject a command, they gradually redirect what the agent believes its objective is, across multiple turns, so no single message looks like an attack.
- **Memory poisoning** — false or malicious data written into the [Stage 4](stage-04-memory-state.md) long-term store persists and re-surfaces in future, unrelated sessions — the injection payload outlives the conversation that planted it.
- **Insecure inter-agent communication** — in a [Stage 8](stage-08-multi-agent.md) multi-agent system, a compromised or malicious agent can feed poisoned state to others that trust it by default (the shared-session model has no built-in notion of "don't trust this sender").
- **Cascading multi-agent failures** — one agent's bad output becomes another agent's bad input becomes a third agent's bad decision; a fault that would be a single bad answer in a single-agent system compounds across a pipeline.
- **Rogue agents** — an agent whose persisted state (session, memory) has been subtly corrupted keeps acting on that corrupted state across restarts, looking healthy from the outside.

None of these are hypothetical add-ons — they're what "prompt injection" turns into once an agent has memory that outlives a conversation and peers it trusts by default. Worth a deliberate pass once Stages 4, 6, and 8 are in place, not just this stage's tool-output sanitization in isolation.

## Testing for these — red-teaming, not just eval

[Stage 10](stage-10-evaluation-qa.md)'s golden-dataset eval catches *regressions on known-good behavior*. It does not catch *an adversarial user actively trying to break the agent* — that's a different exercise: automated red-teaming, where an attacker-LLM generates adversarial multi-turn conversations specifically designed to trigger goal hijacking, memory poisoning, or injection, and every finding that succeeds gets converted into a permanent regression test in the Stage 10 suite so the same attack can never work twice. Treat this stage's "Done when" (a single injection attempt gets neutralized) as the starting bar, not the finish line — a real red-team pass tries many variations, not one.

## Sources

| Type | Resource |
|------|----------|
| Standard | [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — the vendor-neutral reference for tool-use and LLM-application risk; treat this as ground truth over any single provider's guidance |
| Standard | [OWASP Top 10 for Agentic Applications — full guide](https://www.aikido.dev/blog/owasp-top-10-agentic-applications) |
| Guide | [Prompt Injection in 2026 — OWASP's #1 LLM threat](https://www.kunalganglani.com/blog/prompt-injection-2026-owasp-llm-vulnerability) |
| Docs | [Claude Platform Docs — safe tool-use design](https://platform.claude.com/docs/en/build-with-claude/tool-use) — one provider's mitigation guidance on top of the OWASP standard above, not a substitute for it |
| Tool | NeMo Guardrails, LLM-Guard, Llama Guard 3 — off-the-shelf guardrail frameworks |
| Repo | [LLMSecurityGuide — OWASP GenAI risks, red-teaming tool catalog](https://github.com/requie/LLMSecurityGuide) |
| Guide | [Confident AI — best AI red-teaming tools 2026](https://www.confident-ai.com/knowledge-base/compare/best-ai-red-teaming-tools-2026) |

## Done when

A tool result containing "ignore previous instructions and…" gets logged and neutralized, not obeyed — and a red-team pass covering at least goal hijacking and memory poisoning produces zero new successful attacks, or every successful one has become a permanent regression test.

---
[← Stage 11 — Observability + Tracing](stage-11-observability.md) · [Back to roadmap](../README.md) · **Next:** [Stage 13 — Production Deployment →](stage-13-deployment.md)
