"""
The same agent again — this time using LangChain's model wrapper and
@tool decorator, but deliberately WITHOUT LangGraph.

The point of this file is what it does NOT have: no StateGraph, no
ToolNode, no tools_condition, no cycle. LangChain gives you a nicer model
wrapper and automatic tool-schema generation, but it has no primitive for
"loop back and call the model again after the tool result." So the loop
below is hand-written — structurally identical to example 01's raw-API
while loop, just with LangChain's ChatAnthropic instead of the Anthropic
SDK client, and LangChain's @tool instead of a hand-written JSON schema.

Compare all three examples side by side:
  01-basic-agent            raw Anthropic SDK, hand-written loop
  01-basic-agent-langchain  LangChain's ChatAnthropic + @tool, hand-written loop (this file)
  01-basic-agent-langgraph  LangChain's @tool + LangGraph's StateGraph, the loop IS the graph

The middle one is the one people expect to not need a loop for — it still
does. That's the actual difference between LangChain and LangGraph: LangChain
is components (model wrappers, tool schemas, prompt templates); LangGraph is
the graph/cycle engine that removes the hand-written loop. Neither one
"replaces" tool-calling logic on its own — LangGraph does, LangChain doesn't.

Usage:
    Put ANTHROPIC_API_KEY=sk-ant-... in a .env file at the repo root
    (see .env.example), or export it in your shell — either works.
    uv run agent.py "What is 23 * 47, plus 100?"
"""

import ast
import operator
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "claude-haiku-4-5"
MAX_ITERATIONS = 8
SYSTEM_PROMPT = (
    "You are a precise assistant. For any arithmetic, always use "
    "the calculate tool rather than computing it yourself."
)

# --- Tool implementation (identical AST evaluator to example 01) ----------

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


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+, -, *, /, **, parentheses).

    Call this instead of doing math yourself — it's exact, you aren't.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


TOOLS = [calculate]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# LangChain gives you this much for free: bind_tools() attaches the
# @tool-derived schema to the model, same as example 01's `tools=TOOLS`
# kwarg but without hand-writing the JSON Schema.
llm = ChatAnthropic(model=MODEL, max_tokens=1024).bind_tools(TOOLS)


# --- The agent loop — hand-written, same shape as example 01 --------------
# LangChain does NOT provide this loop. There is no LangChain primitive
# that says "call the model, then if it asked for a tool, run the tool and
# call the model again." That control flow is exactly what LangGraph adds
# (see 01-basic-agent-langgraph/agent.py) — here, it's on you to write it,
# just as it was with the raw Anthropic SDK in example 01.

def run_agent(user_task: str) -> str:
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_task)]

    for step in range(MAX_ITERATIONS):
        response: AIMessage = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # Model is done — same check as example 01's
            # `if response.stop_reason != "tool_use"`, expressed against
            # LangChain's AIMessage.tool_calls instead of stop_reason.
            return response.content or "(no text response)"

        # Run every tool call in this turn, appending one ToolMessage per
        # call — LangChain's equivalent of example 01's tool_results list,
        # just modeled as individual messages instead of one batched
        # tool_result content block.
        for call in response.tool_calls:
            print(f"  [tool call] {call['name']}({call['args']})", file=sys.stderr)
            tool_fn = TOOLS_BY_NAME.get(call["name"])
            result = tool_fn.invoke(call["args"]) if tool_fn else f"Error: unknown tool '{call['name']}'"
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    return "(gave up: exceeded MAX_ITERATIONS without a final answer)"


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "What is 23 * 47, plus 100?"
    print(run_agent(task))
