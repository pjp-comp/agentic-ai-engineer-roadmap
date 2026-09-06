"""
A multi-tool AI agent — same ReAct loop as examples/stage03-tool-calling,
but with four tools instead of one, so the interesting part isn't
"can it call a tool" but "does it call the *right* tool."

Tools: calculate, word_count, convert_units, get_weather (a fake
lookup, deliberately included to show the model calling a tool that
can return an error and recovering instead of crashing).

Runs against a free local Ollama model by default (USE_LOCAL_MODEL=true
in .env.example) via the raw `ollama` Python client — see
examples/stage03-tool-calling/agent.py for the same local/Claude split on a
single tool; this file is that same split applied to four.

Usage:
    Local (default): ollama pull llama3.2:3b, then just run it.
    Claude: put ANTHROPIC_API_KEY=sk-ant-... in a .env file at the repo
    root (see .env.example) and set USE_LOCAL_MODEL=false.
    python agent.py "How many words are in 'the quick brown fox'? Also, what's 12 * 7?"
"""

import os
import sys
import ast
import operator
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "true").strip().lower() == "true"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.2:3b")
CLAUDE_MODEL = "claude-haiku-4-5"
MAX_ITERATIONS = 8

# --- Tool implementations -------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Safely evaluate a numeric expression (no eval() — AST whitelist only)."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


def word_count(text: str) -> str:
    """Count words in a string."""
    return str(len(text.split()))


_LENGTH_TO_METERS = {
    "m": 1.0,
    "km": 1000.0,
    "mi": 1609.344,
    "ft": 0.3048,
}


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a length between m, km, mi, ft."""
    # Coerce value to float — some models (notably smaller local ones) don't
    # always respect the JSON schema's "type": "number" and send a string
    # (e.g. '5') instead. Claude follows the schema strictly; not every
    # tool-calling model does, so parse defensively rather than trust it.
    try:
        value = float(value)
    except (TypeError, ValueError):
        return f"Error: '{value}' is not a valid number"
    from_unit, to_unit = from_unit.lower(), to_unit.lower()
    if from_unit not in _LENGTH_TO_METERS or to_unit not in _LENGTH_TO_METERS:
        supported = ", ".join(_LENGTH_TO_METERS)
        return f"Error: unsupported unit — supported units are {supported}"
    meters = value * _LENGTH_TO_METERS[from_unit]
    result = meters / _LENGTH_TO_METERS[to_unit]
    # Round before returning. A tool's output goes straight into the model's
    # context and usually straight into the user's answer, so binary float
    # noise ("3.1068559611866697 miles") leaks all the way through unless the
    # tool trims it here. Formatting is the tool's job, not the model's --
    # asking the prompt to "round nicely" is a much less reliable fix than
    # simply not emitting the noise in the first place.
    return f"{result:.4g}"


# Deliberately tiny and incomplete — the point is to show the model handling
# a tool that can return a real "not found" error, not to be a real API.
_FAKE_WEATHER = {"paris": "18°C, light rain", "tokyo": "27°C, sunny"}


def get_weather(city: str) -> str:
    """Look up weather for a city (fake, fixed data — no network call)."""
    result = _FAKE_WEATHER.get(city.lower())
    if result is None:
        known = ", ".join(c.title() for c in _FAKE_WEATHER)
        return f"Error: no weather data for '{city}' — known cities: {known}"
    return result


TOOLS = [
    {
        "name": "calculate",
        "description": (
            "Evaluate a basic arithmetic expression (+, -, *, /, **, parentheses). "
            "Call this instead of doing math yourself — it's exact, you aren't."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '23 * 47 + 100'"}
            },
            "required": ["expression"],
        },
    },
    {
        "name": "word_count",
        "description": "Count the number of words in a piece of text.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "convert_units",
        "description": (
            "Convert a length between units. Supported units: m, km, mi, ft."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string", "description": "one of: m, km, mi, ft"},
                "to_unit": {"type": "string", "description": "one of: m, km, mi, ft"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Look up the current weather for a city. Only a small fixed set of "
            "cities is known — may return a not-found error."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
]

_DISPATCH = {
    "calculate": lambda i: calculate(i["expression"]),
    "word_count": lambda i: word_count(i["text"]),
    "convert_units": lambda i: convert_units(i["value"], i["from_unit"], i["to_unit"]),
    "get_weather": lambda i: get_weather(i["city"]),
}

SYSTEM_PROMPT = (
    "You are a precise assistant with several tools available. "
    "Pick whichever tool actually fits the task — don't call a tool "
    "you don't need, and don't compute by hand what a tool can do exactly. "
    "If a task requires more than one tool call, call the next tool again "
    "for each remaining step — do not switch to doing the rest yourself "
    "just because you already called a tool once. Only give a final text "
    "answer once every step in the task is done. "
    "When a word problem describes multiple arithmetic steps (e.g. 'do X "
    "and Y, then divide the result by Z'), be careful with order of "
    "operations: 'the result' of an earlier step must be fully computed "
    "— and parenthesized if needed — before the next operation is "
    "applied to it. Do not write one flat expression like 'a*b+c/d' for "
    "a task that means '(a*b+c)/d' — get this wrong and calculate will "
    "compute a different, wrong expression correctly, which is worse "
    "than an obvious error because it looks right."
)


# --- The agent loop, Claude path --------------------------------------------

def _run_agent_claude(user_task: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_task}]

    for step in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Always append the full response content — it may contain both
        # text and tool_use blocks, and dropping either breaks the next turn.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Model is done — return its final text answer.
            return next(
                (b.text for b in response.content if b.type == "text"),
                "(no text response)",
            )

        # Execute every tool call in this turn, then send all results back
        # in a single user message (required when a turn has >1 tool_use).
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  [tool call] {block.name}({block.input})", file=sys.stderr)
            handler = _DISPATCH.get(block.name)
            result = handler(block.input) if handler else f"Error: unknown tool '{block.name}'"
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})

    return "(gave up: exceeded MAX_ITERATIONS without a final answer)"


# --- The agent loop, local (Ollama) path ------------------------------------
# Same loop shape as the Claude path — the only real difference is the SDK's
# tool-call shape (see examples/stage03-tool-calling/agent.py for the single-tool
# version of this same split, with more detailed comments).

_OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
]


_TOOL_NAMES = {t["name"] for t in TOOLS}
_REQUIRED_PARAMS = {t["name"]: set(t["input_schema"]["required"]) for t in TOOLS}


def _extract_written_out_tool_call(text: str) -> tuple[str, dict] | None:
    """If `text` contains a clean, well-formed tool-call JSON blob — the
    model wrote {"name": ..., "parameters": {...}} as text instead of
    actually calling the tool — return (tool_name, arguments). Otherwise
    return None. See examples/stage03-tool-calling/agent.py for the full
    explanation; this is the same idea generalized to four tools instead
    of one — deliberately strict: only fires on a clean parse with a known
    tool name and all its required arguments present. Anything messier is
    left alone and returned as plain text, same as before.
    """
    if '"name"' not in text:
        return None
    import json
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    if not isinstance(name, str) or name not in _TOOL_NAMES:
        return None
    params = parsed.get("parameters")
    if not isinstance(params, dict) or not _REQUIRED_PARAMS[name] <= params.keys():
        return None
    return name, params


def _run_agent_local(user_task: str) -> str:
    import ollama

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_task},
    ]

    for step in range(MAX_ITERATIONS):
        response = ollama.chat(model=LOCAL_MODEL, messages=messages, tools=_OLLAMA_TOOLS)
        msg = response.message

        if not msg.tool_calls:
            written_out = msg.content and _extract_written_out_tool_call(msg.content)
            if written_out:
                # The model wrote a correct-looking tool call as text
                # instead of issuing it for real. Run it directly — no
                # round trip asking the model to "please call it again,"
                # since that risks a different (sometimes worse) response
                # the second time.
                name, args = written_out
                print(f"  [tool call] {name}({args}) — written as text, running directly", file=sys.stderr)
                handler = _DISPATCH.get(name)
                result = handler(args) if handler else f"Error: unknown tool '{name}'"
                messages.append({"role": "assistant", "content": msg.content})
                messages.append({"role": "tool", "tool_name": name, "content": result})
                continue
            return msg.content or "(no text response)"

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

        for call in msg.tool_calls:
            name = call.function.name
            args = call.function.arguments
            print(f"  [tool call] {name}({args})", file=sys.stderr)
            handler = _DISPATCH.get(name)
            result = handler(args) if handler else f"Error: unknown tool '{name}'"
            # tool_name is load-bearing once a turn issues MORE THAN ONE tool
            # call. Three unlabeled {"role": "tool"} messages arrive back as an
            # anonymous list, and the model has to guess which result answers
            # which call -- in practice it guesses wrong and starts answering a
            # question nobody asked. With a single tool you can get away with
            # omitting it; with four you cannot. This is the multi-tool-specific
            # bug this example exists to surface, so it is fixed here rather
            # than left as an exercise.
            messages.append({"role": "tool", "tool_name": name, "content": result})

    return "(gave up: exceeded MAX_ITERATIONS without a final answer)"


def run_agent(user_task: str) -> str:
    if USE_LOCAL_MODEL:
        print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)
        return _run_agent_local(user_task)
    print(f"  [model] Claude API: {CLAUDE_MODEL}", file=sys.stderr)
    return _run_agent_claude(user_task)


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or (
        "How many words are in 'the quick brown fox jumps'? "
        "Also convert 5 km to miles. Also what's the weather in Tokyo?"
    )
    print(run_agent(task))
