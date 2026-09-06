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

## Trajectory evaluation — grading the path, not just the destination

The brief above grades the agent's **final answer**. That's how you evaluate an LLM. Evaluating an *agent* needs a second axis, because an agent that arrives at the right answer the wrong way is a latent bug, not a pass:

- It called an expensive tool eleven times where one call would do — correct, but it will bankrupt you at volume.
- It got the right answer from its training data without ever calling the tool that was supposed to ground it — correct today, wrong the moment the underlying data changes.
- It tried three wrong tools before stumbling onto the right one — correct, but it will fail on a slightly harder variant.
- It looped four times and hit the iteration cap, returning a degraded partial answer that happened to be right.

None of those show up in an outcome-only score. **Trajectory evaluation** grades the sequence of steps itself. The four measures worth building, in increasing order of effort:

| Measure | What it asks | How to score it |
|---|---|---|
| **Tool-selection accuracy** | Did it call the right tool for the step? | Deterministic — compare against an expected tool name; no LLM needed |
| **Step efficiency** | How many steps/tool calls versus the minimum? | Deterministic — count them, compare to a per-case expected count |
| **Trajectory match** | Did the sequence of tool calls match an expected path? | Deterministic for exact match; use "expected calls appear in order, extra calls allowed" for a more forgiving version that doesn't break every time the agent takes a valid alternate route |
| **Trajectory quality** | Was each step *reasonable* given what was known at that point? | LLM judge over the step log — the only one needing a judge, and the one to add last |

Start with the two deterministic ones. They're nearly free — your Stage 11 traces already contain the data — and they catch the two most expensive failure modes (redundant tool calls, and never calling the tool at all). **A good rule: any case in the golden dataset where the agent has a tool it's supposed to use should assert that it used it**, not just that the answer looked right.

The natural pairing here: [Stage 11's tracing](stage-11-observability.md) produces the per-node spans, and this stage turns those spans into assertions. If Stage 11 is built first, trajectory eval is mostly a matter of reading trace records you're already writing.

## Validate the judge before you trust the metric

An LLM judge is itself a model that can be wrong, and a pass rate is only as trustworthy as the judge producing it. The failure that bites hardest is silent: a judge that's systematically lenient reports 95% while the agent is at 70%, and you ship a regression believing you tested for it.

Judges have known, documented biases worth designing against:

- **Self-preference** — a model rates its own outputs higher than another model's. This is why [`examples/stage10-eval-harness/`](../examples/stage10-eval-harness/) judging with a *different* model than the one under test isn't a stylistic choice; it's the mitigation.
- **Position bias** — in a pairwise comparison, the option presented first wins more often than it should. Mitigation: run both orderings and keep only the cases where the verdict is consistent.
- **Verbosity bias** — longer answers get scored higher regardless of correctness.
- **Leniency drift** — judges skew toward passing when a rubric is vague, which is exactly why the brief specifies structured boolean criteria rather than "rate 1–10."

**The validation step itself, once, before you rely on the number:** hand-label 20–30 cases yourself, run the judge over the same cases, and measure agreement. If the judge agrees with you on fewer than ~80% of cases, fix the rubric before fixing the agent — you're currently optimizing against a broken measuring instrument. Re-run this check whenever you change the judge model or the rubric, and keep the hand-labeled set: it's the eval suite *for your eval suite*.

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
