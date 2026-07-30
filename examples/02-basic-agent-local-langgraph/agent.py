"""
The same LangGraph agent as examples/01-basic-agent-langgraph, but able to
run entirely offline against a free, open-weight model via Ollama instead
of the Claude API — no API key, no per-token cost.

Same tool (calculate), same graph shape, same iteration cap. The only
difference from example 01-basic-agent-langgraph is which chat model
gets bound to the graph, chosen at runtime from one .env flag:

    USE_LOCAL_MODEL=true   -> ChatOllama(model=LOCAL_MODEL), runs on your machine
    USE_LOCAL_MODEL=false  -> ChatAnthropic(model=MODEL), calls the Claude API

This is the whole lesson of this example: the graph, the tool, and the
ReAct loop do not care which model answers the "what should I do next"
question. Swapping the model is a one-line change at the edge of the
program, not a rewrite of the agent.

Setup (local model path):
    1. Install Ollama: https://ollama.com/download
    2. ollama pull llama3.2:3b   (~2GB download, one-time)
    3. Set USE_LOCAL_MODEL=true in your repo-root .env (see .env.example)
    4. uv run agent.py "What is 23 * 47, plus 100?"

Setup (Claude API path — default):
    Put ANTHROPIC_API_KEY=sk-ant-... in a .env file at the repo root,
    leave USE_LOCAL_MODEL=false (or unset), then run the same command.
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

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
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

# --- Model selection --------------------------------------------------------
# This is the one part of the file that differs from example
# 01-basic-agent-langgraph: which chat model the graph binds its tools to.
# Everything below this block (the graph, the nodes, the edges) is identical
# regardless of which branch runs, because both ChatAnthropic and ChatOllama
# implement the same LangChain chat-model interface (.bind_tools(), .invoke(),
# returning an AIMessage with .tool_calls).

if USE_LOCAL_MODEL:
    from langchain_ollama import ChatOllama

    print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)
    llm = ChatOllama(model=LOCAL_MODEL, temperature=0).bind_tools(TOOLS)
else:
    from langchain_anthropic import ChatAnthropic

    print(f"  [model] Claude API: {CLAUDE_MODEL}", file=sys.stderr)
    llm = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024).bind_tools(TOOLS)


# --- The graph (identical to example 01-basic-agent-langgraph) -------------

def call_model(state: MessagesState) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = llm.invoke(messages)
    for call in response.tool_calls:
        print(f"  [tool call] {call['name']}({call['args']})", file=sys.stderr)
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()


def run_agent(user_task: str) -> str:
    result = app.invoke(
        {"messages": [{"role": "user", "content": user_task}]},
        config={"recursion_limit": MAX_ITERATIONS * 2},
    )
    final = result["messages"][-1]
    return final.content or "(no text response)"


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "What is 23 * 47, plus 100?"
    print(run_agent(task))
