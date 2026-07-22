[← Stage 10](stage-10-observability.md) · Stage 11 of 13 · **Next:** [Stage 12 →](stage-12-deployment.md)

# Stage 11 — Security + Guardrails

Prompt injection defense · output filtering · PII redaction · sandboxed execution · compliance

## Why this matters

Prompt injection has held the #1 spot on OWASP's LLM Top 10 across every edition. Once an agent has tools with real side effects, untrusted content (a webpage, an email, a file) becomes an attack surface. Assume the model will occasionally be tricked, and design containment — least-privilege tools, output validation, sandboxing — rather than relying on the model to "just not fall for it."

## Brief

Add a guard layer: strip/flag instructions found inside tool-returned content before it re-enters the prompt, enforce a tool allowlist per agent role, and run any code-execution tool in a sandboxed subprocess with no filesystem/network access by default.

```python
def sanitize_tool_output(text: str) -> str:
    if INJECTION_PATTERNS.search(text):
        log_security_event("possible_injection", text)
    return wrap_as_untrusted_data(text)  # tag so the model
    # is instructed to treat it as data, never as instructions
```

## Sources

| Type | Resource |
|------|----------|
| Standard | [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) |
| Guide | [Prompt Injection in 2026 — OWASP's #1 LLM threat](https://www.kunalganglani.com/blog/prompt-injection-2026-owasp-llm-vulnerability) |
| Docs | [Claude Platform Docs — safe tool-use design](https://platform.claude.com/docs/en/build-with-claude/tool-use) |
| Tool | NeMo Guardrails, LLM-Guard, Llama Guard 3 — off-the-shelf guardrail frameworks |
| Repo | [LLMSecurityGuide — OWASP GenAI risks, red-teaming tool catalog](https://github.com/requie/LLMSecurityGuide) |

## Done when

A tool result containing "ignore previous instructions and…" gets logged and neutralized, not obeyed.

---
[← Stage 10 — Observability + Tracing](stage-10-observability.md) · [Back to roadmap](../README.md) · **Next:** [Stage 12 — Production Deployment →](stage-12-deployment.md)
