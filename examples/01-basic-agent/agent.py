"""
A basic AI agent, built from raw API calls — no framework.

One tool (a calculator), one loop: call the model, if it wants a tool,
run the tool and feed the result back, repeat until it answers in
plain text. This is what LangGraph/CrewAI automate for you — seeing
it unframeworked once makes the framework's job legible.

Runs against a free local Ollama model by default (USE_LOCAL_MODEL=true
in .env.example) via the raw `ollama` Python client — same "no framework"
spirit as the Claude path below, just a different SDK. Set
USE_LOCAL_MODEL=false to use the Claude API instead.

Usage:
    Local (default): ollama pull llama3.2:3b, then just run it.
    Claude: put ANTHROPIC_API_KEY=sk-ant-... in a .env file at the repo
    root (see .env.example) and set USE_LOCAL_MODEL=false.
    python agent.py "What is 23 * 47, plus 100?"
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


SYSTEM_PROMPT = (
    "You are a precise assistant. For any arithmetic, always use "
    "the calculate tool rather than computing it yourself."
)


# --- The agent loop, Claude path ------------------------------------------

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
            if block.name == "calculate":
                result = calculate(block.input["expression"])
            else:
                result = f"Error: unknown tool '{block.name}'"
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})

    return "(gave up: exceeded MAX_ITERATIONS without a final answer)"


# --- The agent loop, local (Ollama) path ----------------------------------
# Structurally identical to the Claude path above — same loop, same tool,
# same "check the model's response, run a tool or return text" shape. The
# only difference is the SDK: `ollama.chat()` instead of
# `client.messages.create()`, and its tool-call shape
# (`resp.message.tool_calls`, each a `ToolCall(function=Function(name=...,
# arguments=...))`) instead of Anthropic's `tool_use` content blocks.

_OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOLS[0]["name"],
            "description": TOOLS[0]["description"],
            "parameters": TOOLS[0]["input_schema"],
        },
    }
]


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
            # Model is done — same check as the Claude path's
            # `stop_reason != "tool_use"`, expressed against Ollama's
            # tool_calls list instead of a stop_reason field.
            return msg.content or "(no text response)"

        # Ollama expects the assistant turn (including tool_calls) appended
        # before the tool results, same ordering requirement as Claude.
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

        for call in msg.tool_calls:
            name = call.function.name
            args = call.function.arguments
            print(f"  [tool call] {name}({args})", file=sys.stderr)
            if name == "calculate":
                result = calculate(args["expression"])
            else:
                result = f"Error: unknown tool '{name}'"
            # Ollama's chat API takes tool results back as role="tool"
            # messages, one per call — its equivalent of Claude's batched
            # tool_result content blocks, just modeled as separate messages.
            messages.append({"role": "tool", "content": result})

    return "(gave up: exceeded MAX_ITERATIONS without a final answer)"


def run_agent(user_task: str) -> str:
    if USE_LOCAL_MODEL:
        print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)
        return _run_agent_local(user_task)
    print(f"  [model] Claude API: {CLAUDE_MODEL}", file=sys.stderr)
    return _run_agent_claude(user_task)


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "What is 23 * 47, plus 100?"
    print(run_agent(task))
