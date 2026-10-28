[← Learning path](../../README.md) · [Stage 12 lesson](../../docs/stage-12-security-guardrails.md)

# Tool permissions and untrusted content

This demo separates three ideas: spotting suspicious text, enforcing tool permissions, and accepting only a small arithmetic language. Only the latter two are deterministic controls. Labeling content is not a guarantee that a model ignores injection.

## Run it

```bash
cd examples/stage12-guardrails
ollama pull llama3.2:3b
uv run agent.py
```

Five scenarios show clean order data, injected order data, a denied calculator call, an allowed calculation, and a rejected function call. Orders are synthetic and roles are hard-coded. No authentication or general code-execution sandbox is implemented.

## Read the code

- `sanitize_tool_output()` logs a few recognizable attack patterns and labels content as untrusted. It does not neutralize the content; inspect the generated answer yourself.
- `enforce_allowlist()` rejects unknown roles and unauthorized tool names before the tool runs. Production roles must come from verified identity, not model output.
- `evaluate_arithmetic()` walks an allowlisted syntax tree. It accepts numeric literals, parentheses, and `+ - * / // %`; calls, attributes, names, and exponentiation are rejected. Input length, node count, and numeric magnitude are bounded.
- `run_calculator()` calls the arithmetic interpreter and converts rejected input into a tool error. It does not execute Python code.

Expected deterministic checks: `support` cannot run `calculate`; `admin` can calculate `2+2` as `4`; `__import__('os').listdir('.')` returns an error. The attack scenario should answer the order question without following the injected instruction, but that result must be checked on each model/version.

## Try it yourself

Add an unknown role and verify that no tool executes. Try attribute access such as `(1).__class__`, oversized input, and division by zero. Then rephrase the injection and record whether the simple pattern detector notices it and whether the model follows it. These are separate questions.

Keep model-dependent failures as evaluation cases. For production isolation and approval design, continue to [Stage 13](../../docs/stage-13-deployment.md) and [Stage 09](../../docs/stage-09-human-in-the-loop.md).
