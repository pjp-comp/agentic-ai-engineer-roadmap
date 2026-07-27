"""
A multi-tool AI agent — same ReAct loop as examples/01-basic-agent,
but with four tools instead of one, so the interesting part isn't
"can it call a tool" but "does it call the *right* tool."

Tools: calculate, word_count, convert_units, get_weather (a fake
lookup, deliberately included to show the model calling a tool that
can return an error and recovering instead of crashing).

Usage:
    Put ANTHROPIC_API_KEY=sk-ant-... in a .env file at the repo root
    (see .env.example), or export it in your shell — either works.
    python agent.py "How many words are in 'the quick brown fox'? Also, what's 12 * 7?"
"""

import sys
import ast
import operator
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "claude-haiku-4-5"
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
    from_unit, to_unit = from_unit.lower(), to_unit.lower()
    if from_unit not in _LENGTH_TO_METERS or to_unit not in _LENGTH_TO_METERS:
        supported = ", ".join(_LENGTH_TO_METERS)
        return f"Error: unsupported unit — supported units are {supported}"
    meters = value * _LENGTH_TO_METERS[from_unit]
    return str(meters / _LENGTH_TO_METERS[to_unit])


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


# --- The agent loop --------------------------------------------------------

def run_agent(user_task: str) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_task}]

    for step in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=(
                "You are a precise assistant with several tools available. "
                "Pick whichever tool actually fits the task — don't call a tool "
                "you don't need, and don't compute by hand what a tool can do exactly."
            ),
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


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or (
        "How many words are in 'the quick brown fox jumps'? "
        "Also convert 5 km to miles. Also what's the weather in Tokyo?"
    )
    print(run_agent(task))
