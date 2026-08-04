"""
Eval harness with a golden dataset and an LLM-as-judge (Stage 10 --
docs/stage-10-evaluation-qa.md), matching that stage's brief:

    class JudgeVerdict(BaseModel):
        followed_instructions: bool
        factually_grounded: bool
        reasoning: str

    def judge(expected: str, actual: str) -> JudgeVerdict:
        return llm.parse(...)

Ollama only -- both the agent under test and the judge run locally.
Deliberately two DIFFERENT models: a small one (llama3.2:3b) as the agent
being evaluated, a larger one (qwen3:1.7b, temperature=0, thinking off)
as the judge -- using the same model to grade itself is a well-known way
to get inflated pass rates, so this example doesn't do that even at small
local-model scale.

This is not a framework. It's the smallest real version of Stage 10's
brief: a golden dataset (10 cases, per the "Beginner focus" section's
"start with 10 to 15"), an agent to test, a structured-output judge, and
one headline metric (pass rate) -- run on every call, so a prompt change
that breaks something is visible immediately, the way the "Done when"
section describes.

Usage:
    ollama pull llama3.2:3b   # one-time, ~2GB, agent under test
    ollama pull qwen3:1.7b    # one-time, ~1.4GB, the judge
    uv run agent.py                 # run the full golden dataset, print pass rate
    uv run agent.py --case 3        # run and judge a single case, verbose
    uv run agent.py --baseline 0.8  # exit 1 if pass rate drops below this (CI use)
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import ollama
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

AGENT_MODEL = "llama3.2:3b"   # the system under test
JUDGE_MODEL = "qwen3:1.7b"    # the grader -- deliberately a different model


class JudgeVerdict(BaseModel):
    followed_instructions: bool
    factually_grounded: bool
    reasoning: str

    @property
    def passed(self) -> bool:
        # A single ambiguous "rate 1-10" score is exactly what Stage 10's
        # sources warn against -- pass/fail here is derived from two
        # explicit, separately-judged criteria instead.
        return self.followed_instructions and self.factually_grounded


@dataclass
class GoldenCase:
    id: int
    prompt: str
    expected_behavior: str   # what a correct answer must contain/do -- not a literal string match


# 10 cases, per Stage 10's beginner guidance ("start with 10 to 15 test
# cases"). Deliberately mixed: some are easy, a few are designed to catch
# a specific known failure mode (refusing to guess, unit correctness,
# not fabricating a number) so a regression in agent behavior actually
# shows up as a failing case here instead of passing by accident.
GOLDEN_DATASET = [
    GoldenCase(1, "What is 12 * 8?", "The answer must state 96."),
    GoldenCase(2, "What's the capital of Japan?", "The answer must state Tokyo."),
    GoldenCase(3, "Convert 100 Fahrenheit to Celsius.", "The answer must be approximately 37.8C (within 1 degree)."),
    GoldenCase(4, "Who won the 2019 Nobel Prize in a fictional category called 'Best Sandwich'?",
               "There is no such prize -- the answer must say it doesn't know or that no such prize exists, not invent a winner."),
    GoldenCase(5, "Summarize in one sentence: cats are mammals that often sleep 12-16 hours a day.",
               "The answer must mention both that cats are mammals and the 12-16 hour sleep figure."),
    GoldenCase(6, "What year did World War II end?", "The answer must state 1945."),
    GoldenCase(7, "If a train travels 60 miles in 2 hours, what is its average speed?",
               "The answer must state 30 miles per hour (or mph)."),
    GoldenCase(8, "What is the boiling point of water in Celsius at sea level?", "The answer must state 100C."),
    GoldenCase(9, "Name the CEO of a company called 'Zyrqon Dynamics'.",
               "This company does not exist -- the answer must say it doesn't know or the company is unfamiliar, not invent a name."),
    GoldenCase(10, "What is the chemical symbol for gold?", "The answer must state Au."),
]


def run_agent(prompt: str) -> str:
    """The system under test -- a plain single-turn call, no tools, kept
    simple on purpose so the harness's logic (not the agent's) is what
    this example demonstrates.
    """
    resp = ollama.chat(
        model=AGENT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer briefly and directly. If you don't know something or it doesn't exist, say so plainly instead of guessing.",
            },
            {"role": "user", "content": prompt},
        ],
        think=False,
    )
    return resp.message.content or ""


JUDGE_SYSTEM = """You are a strict evaluator for an AI agent's answers. \
You will be given a task prompt, the expected behavior (not a literal \
string to match, but a description of what a correct answer must do), \
and the agent's actual answer. Judge two things independently:

1. followed_instructions: did the answer address what was actually asked, \
in the expected way (e.g. state "don't know" when the expected behavior \
says it should, rather than inventing an answer)?
2. factually_grounded: is the answer factually correct and not fabricated?

Respond with ONLY a JSON object, no other text, matching exactly this shape:
{"followed_instructions": true or false, "factually_grounded": true or false, "reasoning": "one short sentence"}"""


def judge(prompt: str, expected_behavior: str, actual: str) -> JudgeVerdict:
    resp = ollama.chat(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"Task prompt: {prompt}\nExpected behavior: {expected_behavior}\nAgent's answer: {actual}",
            },
        ],
        think=False,
        format=JudgeVerdict.model_json_schema(),
    )
    return JudgeVerdict.model_validate(json.loads(resp.message.content))


def run_case(case: GoldenCase, verbose: bool = False) -> tuple[bool, JudgeVerdict, str]:
    answer = run_agent(case.prompt)
    verdict = judge(case.prompt, case.expected_behavior, answer)
    if verbose:
        print(f"\n[case {case.id}] {case.prompt}")
        print(f"  expected: {case.expected_behavior}")
        print(f"  answer:   {answer}")
        print(f"  verdict:  followed_instructions={verdict.followed_instructions} "
              f"factually_grounded={verdict.factually_grounded}")
        print(f"  reasoning: {verdict.reasoning}")
        print(f"  {'PASS' if verdict.passed else 'FAIL'}")
    return verdict.passed, verdict, answer


def run_suite(baseline: float | None = None) -> None:
    print(f"  [agent model] {AGENT_MODEL}   [judge model] {JUDGE_MODEL}", file=sys.stderr)
    print(f"Running {len(GOLDEN_DATASET)} golden case(s)...\n")

    results = []
    for case in GOLDEN_DATASET:
        passed, verdict, answer = run_case(case)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] case {case.id}: {case.prompt}")
        if not passed:
            print(f"          reasoning: {verdict.reasoning}")
            print(f"          answer:    {answer}")
        results.append(passed)

    pass_rate = sum(results) / len(results)
    print(f"\nPass rate: {sum(results)}/{len(results)} ({pass_rate:.0%})")

    if baseline is not None:
        if pass_rate < baseline:
            print(f"FAIL: pass rate {pass_rate:.0%} is below baseline {baseline:.0%}")
            sys.exit(1)
        print(f"OK: pass rate {pass_rate:.0%} meets baseline {baseline:.0%}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--case":
        case_id = int(argv[1])
        case = next((c for c in GOLDEN_DATASET if c.id == case_id), None)
        if case is None:
            print(f"No case with id={case_id}. Valid ids: 1-{len(GOLDEN_DATASET)}")
            sys.exit(1)
        print(f"  [agent model] {AGENT_MODEL}   [judge model] {JUDGE_MODEL}", file=sys.stderr)
        run_case(case, verbose=True)
    elif argv and argv[0] == "--baseline":
        run_suite(baseline=float(argv[1]))
    else:
        run_suite()
