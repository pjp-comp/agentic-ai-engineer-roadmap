"""
The same news-research agent as examples/01_web_agent_langgraph, but using
Tavily as the search tool instead of Claude's built-in web_search.

The point of this file is the contrast with the other example: Tavily is a
CLIENT-SIDE tool — a real HTTP call that happens in *your* process. That
brings back the ToolNode / tool-dispatch step that Claude's server-side
web_search didn't need, so this graph looks like the calculator examples
(01-basic-agent-langgraph) again: agent -> tools -> agent, not the single
agent-only node from 01_web_agent_langgraph.

    Server-side (Claude's web_search):  agent -----------------------> END
    Client-side (Tavily, this file):    agent -> tools -> agent -> ... -> END

Same task, same system prompt, same "only report what you found" rule —
different tool, different graph shape.

Usage:
    Put ANTHROPIC_API_KEY=sk-ant-... and TAVILY_API_KEY=tvly-... in a .env
    file at the repo root (see .env.example), or export them in your shell.
    uv run agent.py "Polycab India Limited"
    uv run agent.py "Polycab India Limited" --days 7
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "claude-opus-4-8"
MAX_ITERATIONS = 8

SYSTEM_PROMPT = """\
You are a financial news research assistant. Given a company name, use the \
tavily_search tool to find recent news about that company, then write a \
concise, well-organized summary.

Rules:
- Only summarize news you actually found via tavily_search — never invent \
  headlines or events. If search turns up little or nothing for the requested \
  window, say so plainly instead of padding the summary.
- Prefer results with visible dates; note the date next to each item.
- Group related items (e.g. "Earnings", "Regulatory", "Market/analyst commentary") \
  rather than listing headlines in arrival order.
- This is informational summarization, not investment advice — do not add buy/sell \
  recommendations or price targets.
"""

# Tavily is a client-side tool: LangGraph's ToolNode runs the actual HTTP
# call against Tavily's API in this process. max_results caps how much
# content one search pulls back per query (the recall/cost knob); it's not
# the same as capping how many *searches* the model can run (that's why
# MAX_ITERATIONS still exists here, unlike the web_search example).
search_tool = TavilySearch(max_results=5, topic="news")
TOOLS = [search_tool]

llm = ChatAnthropic(model=MODEL, max_tokens=4096).bind_tools(TOOLS)


def call_model(state: MessagesState) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = llm.invoke(messages)
    for call in response.tool_calls:
        print(f"  [tavily_search] {call['args'].get('query')}", file=sys.stderr)
    return {"messages": [response]}


# Same two-node shape as 01-basic-agent-langgraph: the tool is client-side,
# so LangGraph has to actually run it (ToolNode) and loop back to the model
# with the result (tools_condition) before an answer can come back.
graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()


def research_company(company: str, days: int) -> str:
    window_start = date.today() - timedelta(days=days)
    task = (
        f"Find and summarize news about \"{company}\" from the last {days} days "
        f"(since {window_start.isoformat()})."
    )
    result = app.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": MAX_ITERATIONS * 2},
    )

    final = result["messages"][-1]
    if isinstance(final.content, str):
        return final.content or "(no text response)"
    text_parts = [b["text"] for b in final.content if isinstance(b, dict) and b.get("type") == "text"]
    return "".join(text_parts) or "(no text response)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize recent news for a company (via Tavily).")
    parser.add_argument("company", nargs="*", default=["Polycab", "India", "Limited"])
    parser.add_argument("--days", type=int, default=7, help="How many days back to search (default: 7)")
    parsed = parser.parse_args()

    print(research_company(" ".join(parsed.company), parsed.days))
