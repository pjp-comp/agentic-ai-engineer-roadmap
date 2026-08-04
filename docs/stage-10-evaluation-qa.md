[← Stage 09](stage-09-human-in-the-loop.md) · Stage 10 of 14 · **Next:** [Stage 11 →](stage-11-observability.md)

# Stage 10 — Evaluation + Quality Assurance

Automated eval harnesses · LLM-as-a-judge · regression testing · hallucination metrics · named agent benchmarks

## Why this matters

Unlike deterministic software, agent behavior drifts as prompts, models, and tools change. Without a fixed eval set and a repeatable scoring method, you can't tell a genuine improvement from noise — or catch a regression before a user does.

## Beginner focus

- Start with 10 to 15 test cases.
- Use a small, clear rubric.
- Track one headline metric: pass rate.
- Add complex metrics after baseline stability.

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

**Runnable version:** [`examples/stage10-eval-harness/`](../examples/stage10-eval-harness/) builds this exact `JudgeVerdict` shape against a 10-case golden dataset, using **two different local models** — one under test, one judging — so the pass rate isn't inflated by a model grading its own answers. Includes a `--baseline` flag for CI use. Runs entirely on Ollama.

## Sources

| Type | Resource |
|------|----------|
| Docs | [LangSmith — Evaluation framework](https://docs.langchain.com/langsmith/evaluation) |
| Tool | [Arize Phoenix — open-source eval & tracing](https://github.com/Arize-ai/phoenix) |
| Guide | [Hands-on comparison: LangSmith, Langfuse, Arize evals](https://www.analyticsvidhya.com/blog/2026/06/agent-observability-with-langsmith-langfuse-arize/) |
| Concept | Search "LLM-as-judge rubric design" — avoid single-score judges, use structured multi-criteria rubrics |
| Survey | [LangChain — State of Agent Engineering (2025)](https://www.langchain.com/state-of-agent-engineering) — source for the 89%/52% observability-vs-eval gap above |
| Benchmark | [Terminal-Bench](https://www.tbench.ai/) — hand-verified terminal/coding-agent tasks, Stanford/Laude Institute |
| Benchmark | [Recovery-Bench (Letta)](https://github.com/letta-ai/recovery-bench) — measures recovery from corrupted/erroneous context, not just task success |

## The observability/eval gap is real, and it's wide

LangChain's "State of Agent Engineering" survey (Nov–Dec 2025, 1,340 respondents) found 89% of teams running production agents have implemented some form of observability, but only ~52% run offline evals — a 37-point gap between "we can see what the agent did" and "we systematically check whether it did the right thing." Watching an agent isn't the same as testing it; this stage's golden-dataset brief is specifically the less-adopted half of that pair, worth prioritizing precisely because it's the one most teams skip.

## Named benchmarks worth knowing (not building)

The golden dataset in this stage's brief is *your* eval suite, specific to your agent. Separately, a small set of standardized, cross-project benchmarks exist for comparing general agent capability — useful for context when reading about a new model or framework, not something you build yourself:

- **Terminal-Bench** — a Stanford/Laude Institute benchmark of hand-verified terminal/coding tasks, now with a public leaderboard; the reference point for "how good is this agent at real command-line work."
- **Recovery-Bench** — published by Letta, measures whether an agent can recover gracefully from corrupted or erroneous context mid-task, rather than just measuring whether it succeeds when everything goes right.

Treat these the way you'd treat any published leaderboard: informative for comparing models/frameworks in the abstract, but not a substitute for the golden dataset in this stage's brief — that one has to reflect *your* agent's actual task, not a generic benchmark's.

## Eval vs. red-teaming — two different questions

This stage's golden dataset answers "does the agent still behave correctly on cases we already know about?" — a regression check against *known-good* behavior. It does not answer "can an adversarial user find a way to break it?" — that's a distinct discipline (automated red-teaming, adversarial multi-turn simulation) covered in [Stage 12](stage-12-security-guardrails.md#testing-for-these--red-teaming-not-just-eval), because the risks it's testing for (goal hijacking, memory poisoning) are security concerns, not correctness regressions. The two feed each other: every red-team finding that succeeds should become a new case in *this* stage's golden dataset, so the same attack can never work twice without being caught.

## Done when

A prompt change that breaks something gets caught by your eval suite before a human notices.

---
[← Stage 09 — Human-in-the-Loop Systems](stage-09-human-in-the-loop.md) · [Back to roadmap](../README.md) · **Next:** [Stage 11 — Observability + Tracing →](stage-11-observability.md)
