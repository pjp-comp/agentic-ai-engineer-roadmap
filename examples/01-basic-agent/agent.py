"""
A basic AI agent, built from raw API calls — no framework.

One tool (a calculator), one loop: call the model, if it wants a tool,
run the tool and feed the result back, repeat until it answers in
plain text. This is what LangGraph/CrewAI automate for you — seeing
it unframeworked once makes the framework's job legible.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python agent.py "What is 23 * 47, plus 100?"
"""

import sys
import ast
import operator

import anthropic

MODEL = "claude-opus-4-8"
MAX_ITERATIONS = 8

# --- Tool implementation -----------------------------------------------

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
                "expression": {
                    "type": "string",
                    "description": "e.g. '23 * 47 + 100'",
                }
            },
            "required": ["expression"],
        },
    }
]


# --- The agent loop ------------------------------------------------------

def run_agent(user_task: str) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_task}]

    for step in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=(
                "You are a precise assistant. For any arithmetic, always use "
                "the calculate tool rather than computing it yourself."
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
            if block.name == "calculate":
                result = calculate(block.input["expression"])
            else:
                result = f"Error: unknown tool '{block.name}'"
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})

    return "(gave up: exceeded MAX_ITERATIONS without a final answer)"


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "What is 23 * 47, plus 100?"
    print(run_agent(task))
