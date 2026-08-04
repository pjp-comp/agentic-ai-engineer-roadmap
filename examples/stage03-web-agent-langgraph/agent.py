"""
A news research agent: given a company name, it searches the web for recent
news, reads what comes back, and writes a summary — built on LangGraph.

This is a genuinely different shape from the calculator examples (stage03-tool-calling,
stage03-tool-calling-langchain, stage03-tool-calling-langgraph): the tool here is Claude's
built-in `web_search` — a SERVER-SIDE tool. Anthropic's infrastructure runs the
actual search and hands the results back inline in the same API response. There
is no local Python function to execute, so there's no ToolNode / tool-dispatch
step in this graph at all — just one node that calls the model, which is
allowed to call web_search itself, as many times as it needs, before answering.

Claude-only, deliberately, unlike most other examples in this repo:
server-side web_search is a hosted capability of Anthropic's infrastructure,
not something Ollama or any local model can run — there is no local
equivalent to fall back to, so USE_LOCAL_MODEL doesn't apply here. If you
want the same task (company news research) against a local model, see
stage03-web-agent-tavily — Tavily is a client-side tool, so any
tool-calling model (local or Claude) can drive it.

Usage:
    Put ANTHROPIC_API_KEY=sk-ant-... in a .env file at the repo root
    (see .env.example), or export it in your shell — either works.
    uv run agent.py "Polycab India Limited"
    uv run agent.py "Polycab India Limited" --days 7
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """\
You are a financial news research assistant. Given a company name, use the \
web_search tool to find recent news about that company, then write a concise, \
well-organized summary.

Rules:
- Only summarize news you actually found via web_search — never invent headlines \
  or events. If search turns up little or nothing for the requested window, say \
  so plainly instead of padding the summary.
- Prefer results with visible dates; note the date next to each item.
- Group related items (e.g. "Earnings", "Regulatory", "Market/analyst commentary") \
  rather than listing headlines in arrival order.
- This is informational summarization, not investment advice — do not add buy/sell \
  recommendations or price targets.
"""

# Claude's native web_search tool — server-side, no local implementation.
# `max_uses` caps how many searches the model can run in one turn (a cost/
# runaway-loop guard, same purpose as MAX_ITERATIONS in the other examples,
# just enforced server-side instead of by a client loop counter).
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}

llm = ChatAnthropic(model=MODEL, max_tokens=4096).bind_tools([WEB_SEARCH_TOOL])


def call_model(state: MessagesState) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# One node. No ToolNode, no conditional edge back to itself — the model
# resolves web_search calls on Anthropic's side within a single invoke() and
# returns once it's done searching, same as any other single LLM call.
graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

app = graph.compile()


def research_company(company: str, days: int) -> str:
    window_start = date.today() - timedelta(days=days)
    task = (
        f"Find and summarize news about \"{company}\" from the last {days} days "
        f"(since {window_start.isoformat()})."
    )
    result = app.invoke({"messages": [{"role": "user", "content": task}]})

    # Log every search the model actually ran, and surface search errors —
    # web_search failures return a normal 200 with an error object in
    # .content, they do not raise, so silently ignoring them would hide a
    # rate-limit or domain-block problem behind an empty-looking summary.
    final = result["messages"][-1]
    for msg in result["messages"]:
        for block in getattr(msg, "content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "server_tool_use" and block.get("name") == "web_search":
                print(f"  [web_search] {block.get('input', {}).get('query')}", file=sys.stderr)
            if block.get("type") == "web_search_tool_result":
                content = block.get("content")
                if isinstance(content, dict) and "error_code" in content:
                    print(f"  [web_search error] {content['error_code']}", file=sys.stderr)

    # final.content is a plain string only when the response is a single text
    # block. With web_search in play it's usually a list of content blocks
    # (citations split text into multiple segments) — join just the text.
    if isinstance(final.content, str):
        return final.content or "(no text response)"
    text_parts = [b["text"] for b in final.content if isinstance(b, dict) and b.get("type") == "text"]
    return "".join(text_parts) or "(no text response)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize recent news for a company.")
    parser.add_argument("company", nargs="*", default=["Polycab India Limited"])
    parser.add_argument("--days", type=int, default=7, help="How many days back to search (default: 7)")
    parsed = parser.parse_args()

    print(research_company(" ".join(parsed.company), parsed.days))
