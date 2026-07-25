"""
The same agent loop as examples/01-basic-agent, but the tool comes from an
MCP server (server.py) instead of a hard-coded schema + Python function.

Compare this file to 01-basic-agent/agent.py line by line: the ReAct loop
(call model -> if tool_use, run it, feed result back -> repeat) is
IDENTICAL. The only thing that changed is where the tool's schema and
implementation live — this file no longer knows what "calculate" does or
what its input looks like. It discovers that from the MCP server at
startup via list_tools(), and dispatches calls via call_tool() instead of
calling a local Python function directly.

That's the whole point of MCP: the tool became reusable across any MCP
client without this file changing if the tool's implementation changes.

Usage:
    Put ANTHROPIC_API_KEY=sk-ant-... in a .env file at the repo root
    (see .env.example), or export it in your shell — either works.
    uv run agent.py "What is 23 * 47, plus 100?"

    Add --cwu (confirm with user) to pause before every tool call and ask
    for approval in the terminal — a minimal, single-script version of the
    Stage 09 human-in-the-loop pattern (approve/reject/edit a tool call
    before it runs), without the checkpoint/resume machinery a real
    interrupt-and-resume graph needs:
    uv run agent.py --cwu "What is 23 * 47, plus 100?"
"""

import asyncio
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "claude-opus-4-8"
MAX_ITERATIONS = 8


def _mcp_tool_to_claude_schema(tool) -> dict:
    """Convert an MCP tool definition to the shape client.messages.create expects."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema,
    }


def _confirm_tool_call(name: str, args: dict) -> bool:
    """Ask the user in the terminal whether a tool call should run. Blocks on input()."""
    reply = input(f"  [confirm] run {name}({args})? [y/N] ").strip().lower()
    return reply in ("y", "yes")


def _print_token_summary(total_input_tokens: int, total_output_tokens: int) -> None:
    print(
        f"[agent] total tokens: input={total_input_tokens} output={total_output_tokens} "
        f"combined={total_input_tokens + total_output_tokens}",
        file=sys.stderr,
    )


async def run_agent(user_task: str, confirm_with_user: bool = False) -> str:
    client = anthropic.Anthropic()

    server_params = StdioServerParameters(command="uv", args=["run", "server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools from the MCP server instead of hard-coding a schema.
            mcp_tools = (await session.list_tools()).tools
            tools = [_mcp_tool_to_claude_schema(t) for t in mcp_tools]
            print(f"[agent] discovered tools: {tools}", file=sys.stderr)

            messages = [{"role": "user", "content": user_task}]
            total_input_tokens = 0
            total_output_tokens = 0

            for _step in range(MAX_ITERATIONS):
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=(
                        "You are a precise assistant. For any arithmetic, always "
                        "use the calculate tool rather than computing it yourself."
                    ),
                    tools=tools,
                    messages=messages,
                )

                step_in = response.usage.input_tokens
                step_out = response.usage.output_tokens
                total_input_tokens += step_in
                total_output_tokens += step_out
                print(
                    f"[agent] step {_step} tokens: input={step_in} output={step_out} "
                    f"(running total: input={total_input_tokens} output={total_output_tokens})",
                    file=sys.stderr,
                )

                messages.append({"role": "assistant", "content": response.content})
                print(f"[agent] step {_step}: stop_reason={response.stop_reason} content={response.content}", file=sys.stderr)

                if response.stop_reason != "tool_use":
                    final_text = next(
                        (b.text for b in response.content if b.type == "text"),
                        "(no text response)",
                    )
                    _print_token_summary(total_input_tokens, total_output_tokens)
                    return final_text

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    if confirm_with_user and not _confirm_tool_call(block.name, block.input):
                        print(f"  [mcp tool call] {block.name}({block.input}) — rejected by user", file=sys.stderr)
                        content = "Tool call rejected by user. Do not retry this exact call; ask the user what they'd like instead."
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": content,
                                "is_error": True,
                            }
                        )
                        continue

                    print(f"  [mcp tool call] {block.name}({block.input})", file=sys.stderr)
                    # Dispatch via the MCP session instead of calling a local
                    # function directly — this is the one structural change
                    # from 01-basic-agent's loop.
                    result = await session.call_tool(block.name, block.input)
                    content = "".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )
                    print(f"  [mcp tool result] isError={result.isError} content={content!r}", file=sys.stderr)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                            "is_error": result.isError or False,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})

            _print_token_summary(total_input_tokens, total_output_tokens)
            return "(gave up: exceeded MAX_ITERATIONS without a final answer)"


if __name__ == "__main__":
    argv = sys.argv[1:]
    cwu = "--cwu" in argv
    if cwu:
        argv.remove("--cwu")
    task = " ".join(argv) or "What is 23 * 47, plus 100?"
    print(asyncio.run(run_agent(task, confirm_with_user=cwu)))
