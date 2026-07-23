"""
An MCP server exposing one tool: the same calculator from example 01.

This is the same capability as examples/01-basic-agent's `calculate` tool,
now behind the Model Context Protocol instead of a hard-coded schema in
agent.py. Any MCP client — this repo's agent.py, Claude Desktop, a
teammate's agent — can discover and call it without re-implementing
the wrapper.

Run standalone to sanity-check it speaks the protocol:
    uv run server.py
(it will sit waiting for an MCP client on stdio — Ctrl+C to exit)
"""

import ast
import operator

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator-server")

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


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+, -, *, /, **, parentheses).

    Call this instead of doing math yourself — it's exact, you aren't.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


if __name__ == "__main__":
    mcp.run(transport="stdio")
