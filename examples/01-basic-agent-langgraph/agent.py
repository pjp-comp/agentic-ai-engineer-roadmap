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

Runs against a free local Ollama model by default (USE_LOCAL_MODEL=true
in .env.example) via LangChain's ChatOllama. Set USE_LOCAL_MODEL=false to
use the Claude API instead — the graph never branches on which one it got,
same lesson as example 02-basic-agent-local-langgraph.

Usage:
    Local (default): ollama pull llama3.2:3b, then just run it.
    Claude: put ANTHROPIC_API_KEY=sk-ant-... in a .env file at the repo
    root (see .env.example) and set USE_LOCAL_MODEL=false.
    uv run agent.py "What is 23 * 47, plus 100?"
"""

import ast
import operator
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "true").strip().lower() == "true"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.2:3b")
CLAUDE_MODEL = "claude-haiku-4-5"
MAX_ITERATIONS = 8
SYSTEM_PROMPT = (
    "You are a precise assistant. For any arithmetic, always use "
    "the calculate tool rather than computing it yourself."
)

# --- Tool implementation (identical AST evaluator to example 01) ----------
#
# calculate() must never call eval() — the input string comes straight from
# the model, and eval() would let it run arbitrary Python (e.g. "__import__
# ('os').system('rm -rf /')"). Instead we parse the string into an AST and
# walk it ourselves, only ever executing the handful of node types below.
# Anything else (function calls, attribute access, imports, ...) raises
# before it's touched.

# Maps each AST operator node type to the actual Python function that
# performs it. This is the whitelist: if an operator isn't a key in this
# dict, _eval_node() below refuses to run it.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,  # unary minus, e.g. the "-" in "-5"
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate one node of a parsed expression tree.

    `node` is a piece of the AST produced by ast.parse(expr, mode="eval")
    — e.g. for "2 + 3 * 4" the top node is a BinOp("+") whose right side is
    itself a BinOp("*"). This function walks that tree depth-first: each
    call handles one node and recurses into its children, so the whole
    expression is reduced to a single number one operator at a time.

    Three cases, each returning early:
      1. A bare number ("2", "3.5") -> ast.Constant -> return it directly.
      2. A binary operation ("a + b") -> ast.BinOp -> recursively evaluate
         both sides, then apply the matching function from _OPS.
      3. A unary operation ("-a") -> ast.UnaryOp -> recursively evaluate
         the one operand, then apply the matching function from _OPS.

    Anything that isn't one of these three (a function call, a variable
    name, a string, ...) falls through to the raise at the bottom — this
    is what makes the evaluator safe. There's no case that executes
    arbitrary code, so there's nothing for malicious input to hijack.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        # Recurse into left and right before combining — this is what
        # makes nested expressions like "(2 + 3) * 4" work: the inner
        # "2 + 3" is fully resolved to 5 before the outer "* 4" runs.
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    # Reached only for node types we don't explicitly allow above —
    # e.g. ast.Call (a function call) or ast.Name (a variable reference).
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+, -, *, /, **, parentheses).

    Call this instead of doing math yourself — it's exact, you aren't.
    """
    try:
        # mode="eval" parses a single expression (not statements) into a
        # tree whose root is an ast.Expression wrapping the real content
        # in .body — that .body is what we hand to _eval_node().
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        # Any parse error or unsupported node lands here — returned as a
        # normal string, not raised, so the model sees the failure as a
        # tool result it can react to (e.g. retry with a fixed expression)
        # instead of the whole agent crashing.
        return f"Error: could not evaluate '{expression}' ({e})"


TOOLS = [calculate]

# --- The graph -------------------------------------------------------------
# This replaces run_agent()'s `for step in range(MAX_ITERATIONS): ...` loop.
# Two nodes, one conditional edge:
#
#   START -> agent -> (tool call requested?) -> tools -> agent -> ... -> END
#                   -> (no) --------------------------------------------> END

def _build_llm():
    if USE_LOCAL_MODEL:
        from langchain_ollama import ChatOllama

        print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)
        return ChatOllama(model=LOCAL_MODEL, temperature=0).bind_tools(TOOLS)
    from langchain_anthropic import ChatAnthropic

    print(f"  [model] Claude API: {CLAUDE_MODEL}", file=sys.stderr)
    return ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024).bind_tools(TOOLS)


llm = _build_llm()


def call_model(state: MessagesState) -> dict:
    """The 'agent' node — one call to the model, same as one iteration of
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
