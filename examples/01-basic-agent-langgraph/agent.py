"""
The same agent as examples/01-basic-agent, rebuilt on LangGraph instead of
a hand-written while loop.

Same tool (calculate), same task, same iteration cap. What's different is
what LangGraph is doing for you under the hood:

  - MessagesState          replaces the `messages` list you managed by hand.
  - @tool                  replaces the hand-written TOOLS schema dict —
                            LangGraph derives the JSON schema from the
                            function's type hints and docstring.
  - ToolNode                replaces the manual "for block in response.content:
                            if block.type == 'tool_use': ..." dispatch loop.
  - tools_condition          replaces the `if response.stop_reason != "tool_use"`
                            branch that decided whether to keep looping.
  - StateGraph + add_edge   replaces the `for step in range(MAX_ITERATIONS)`
                            loop itself — the graph's edges define the loop.

Read examples/01-basic-agent/agent.py first. Every concept here has a
one-to-one line you already read there; this file just names the pattern
LangGraph gives you instead of writing it by hand.

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
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "claude-opus-4-8"
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

# --- The graph -------------------------------------------------------------
# This replaces run_agent()'s `for step in range(MAX_ITERATIONS): ...` loop.
# Two nodes, one conditional edge:
#
#   START -> agent -> (tool call requested?) -> tools -> agent -> ... -> END
#                   -> (no) --------------------------------------------> END

llm = ChatAnthropic(model=MODEL, max_tokens=1024).bind_tools(TOOLS)


def call_model(state: MessagesState) -> dict:
    """The 'agent' node — one call to Claude, same as one iteration of
    example 01's for-loop body before the tool-dispatch branch."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = llm.invoke(messages)
    for call in response.tool_calls:
        print(f"  [tool call] {call['name']}({call['args']})", file=sys.stderr)
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "agent")
# tools_condition reads the last message's tool_calls and routes to "tools"
# if present, "__end__" otherwise — this is the `stop_reason != "tool_use"`
# check from example 01, expressed as a graph edge instead of an if-statement.
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()


def run_agent(user_task: str) -> str:
    result = app.invoke(
        {"messages": [{"role": "user", "content": user_task}]},
        config={"recursion_limit": MAX_ITERATIONS * 2},  # each loop = 2 graph steps
    )
    final = result["messages"][-1]
    return final.content or "(no text response)"


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "What is 23 * 47, plus 100?"
    print(run_agent(task))
