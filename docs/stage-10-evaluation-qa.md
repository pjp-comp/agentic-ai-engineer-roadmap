[← Stage 09](stage-09-human-in-the-loop.md) · Stage 10 of 14 · **Next:** [Stage 11 →](stage-11-observability.md)

# Stage 10 — Evaluation + Quality Assurance

Automated eval harnesses · LLM-as-a-judge · regression testing · hallucination metrics

## Why this matters

Unlike deterministic software, agent behavior drifts as prompts, models, and tools change. Without a fixed eval set and a repeatable scoring method, you can't tell a genuine improvement from noise — or catch a regression before a user does.

## Brief

Build a golden dataset of 30–50 input/expected-behavior pairs. Run them against the agent on every change, score with a rubric-based LLM judge (not vague "rate 1-10"), and fail CI if the pass rate drops below baseline.

```python
class JudgeVerdict(BaseModel):
    followed_instructions: bool
    factually_grounded: bool
    reasoning: str

def judge(expected: str, actual: str) -> JudgeVerdict:
    return llm.parse(
        prompt=RUBRIC.format(expected=expected, actual=actual),
        output_format=JudgeVerdict,
    )
```

## Sources

| Type | Resource |
|------|----------|
| Docs | [LangSmith — Evaluation framework](https://docs.langchain.com/langsmith/evaluation) |
| Tool | [Arize Phoenix — open-source eval & tracing](https://github.com/Arize-ai/phoenix) |
| Guide | [Hands-on comparison: LangSmith, Langfuse, Arize evals](https://www.analyticsvidhya.com/blog/2026/06/agent-observability-with-langsmith-langfuse-arize/) |
| Concept | Search "LLM-as-judge rubric design" — avoid single-score judges, use structured multi-criteria rubrics |

## Done when

A prompt change that breaks something gets caught by your eval suite before a human notices.

---
[← Stage 09 — Human-in-the-Loop Systems](stage-09-human-in-the-loop.md) · [Back to roadmap](../README.md) · **Next:** [Stage 11 — Observability + Tracing →](stage-11-observability.md)
