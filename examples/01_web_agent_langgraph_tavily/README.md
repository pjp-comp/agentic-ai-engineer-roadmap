[← Back to roadmap](../../README.md)

# Example 01e — The Same Web Research Agent, Using Tavily

Same task as [`01_web_agent_langgraph`](../01_web_agent_langgraph/) — give it a company name, get a dated, grouped news summary — but the search now goes through [Tavily](https://tavily.com/), a search API built specifically for LLM agents, instead of Claude's built-in `web_search`.

```bash
uv run agent.py "polycab india limited" --days 7
```

## What's different about this one

The other example's whole point was that `web_search` is a **server-side** tool — Anthropic's infrastructure runs the search, so there's no `ToolNode` in that graph at all. Tavily is the opposite: it's a **client-side** tool. `TavilySearch` makes a real HTTP call to Tavily's API from *your* process, and LangGraph has to actually run it and route the result back to the model. That brings the two-node shape back:

| | `01_web_agent_langgraph` (Claude `web_search`) | This example (Tavily) |
|---|---|---|
| Tool location | Server-side — Anthropic's infrastructure | Client-side — an HTTP call in your process |
| Graph nodes | `agent` only | `agent` + `tools` (`ToolNode`) |
| Who runs the search | Anthropic | `ToolNode`, via `langchain_tavily.TavilySearch` |
| Loop needed? | No — the model already has results before replying | Yes — `tools_condition` routes back to `agent` after each search, same as the calculator examples |
| Iteration cap | `max_uses` on the tool definition (server enforces it) | `MAX_ITERATIONS` / `recursion_limit`, same pattern as [`01-basic-agent-langgraph`](../01-basic-agent-langgraph/) |

Same task, same system prompt, same "only report what you actually found" rule — different tool, different graph. Read this file right after the other one and the shape difference is the whole lesson: **whether a tool is client-side or server-side determines whether your graph needs a `ToolNode` at all**, not whether the tool is "for search" specifically.

## Why reach for Tavily instead of Claude's `web_search`

Both work; the choice is about what you need control over:

- **Claude's `web_search`** — zero extra setup (no separate API key/vendor), Anthropic handles ranking and fetching. Good default when you don't need to tune the search provider itself.
- **Tavily** — a dedicated `TAVILY_API_KEY`, but purpose-built for agent use: structured JSON results, a `topic="news"` mode (used here), date-scoped queries, and it's swappable independently of which LLM you're using — the same `TavilySearch` tool works with any LangChain-compatible model, not just Claude.

If your project is LangGraph-first and might swap models later, a client-side search tool like Tavily keeps that swap clean. If you're committed to Claude specifically and want the simplest setup, `web_search` needs nothing extra.

## What it is

- **One tool**: `TavilySearch(max_results=5, topic="news")` from `langchain_tavily` — a real API call, not a server-side Anthropic tool. `topic="news"` biases results toward news content specifically (vs. general web results).
- **A two-node graph**: `agent` (calls Claude) and `tools` (`ToolNode` running `TavilySearch`), wired with `tools_condition` — identical wiring to [`01-basic-agent-langgraph`](../01-basic-agent-langgraph/), just with a real search tool instead of the calculator.
- **The same rules-heavy system prompt** as the other web-research example: only summarize what was found, group by category, no fabricated headlines, no investment advice.

## Run it

```bash
cd examples/01_web_agent_langgraph_tavily
uv run agent.py "polycab india limited" --days 7
```

This example needs **two** keys in the repo-root `.env` (see [`.env.example`](../../.env.example)): `ANTHROPIC_API_KEY` (for the model) and `TAVILY_API_KEY` (for search — get one at [tavily.com](https://tavily.com/)). Both load automatically via `python-dotenv` — no `export` needed.

`--days` controls the requested news window (default 7).

Expected output (search queries print to stderr so stdout stays clean prose):

```
  [tavily_search] Polycab India Limited news
  [tavily_search] Polycab India Q1 results earnings stock
  [tavily_search] Polycab India FMEG solar segment growth Project Spring FY27
  [tavily_search] Polycab India brokerage rating target price analyst
```

```
## Earnings — Record Q1 FY27 (reported ~16–17 Jul 2026)
Polycab posted its highest-ever first-quarter figures (consolidated, YoY):
- Revenue: up 39% to ₹8,210 crore; beat the CNBC-TV18 poll estimate.
- Net profit: up ~33% to ₹797 crore.
- EBITDA: up 32% to ₹1,136 crore; EBITDA margin 13.8%.

## Market / Share-price reaction
Despite the earnings beat, the stock fell ~4% on 17 Jul, extending losses for
a third straight session, attributed to profit-booking after a ~48% rally
over the prior three months.

## Analyst / Brokerage commentary
Jefferies, HSBC, and Equirus all maintained Buy/Add ratings post-results,
with price targets in the ₹10,160–10,700 range.

Notes: most reporting clustered on 17 July, at the start of the window;
no material new Polycab-specific developments found in the following days.
This is an informational summary only, not investment advice.
```

Every figure above traces to a specific dated source Tavily returned — the agent cites publications and dates inline, and explicitly flags where sources disagreed (e.g. differing FMEG growth percentages across outlets) rather than picking one silently.

## What to look at closely

- **`ToolNode(TOOLS)` is back** — compare directly to its absence in `01_web_agent_langgraph/agent.py`. This is the concrete answer to "when do I need a ToolNode": whenever the tool executes in your process, which is true for essentially every third-party API you integrate yourself, and false only for Anthropic's own server-side tools (`web_search`, `web_fetch`, `code_execution`).
- **`MAX_ITERATIONS * 2` recursion limit** — same reasoning as the calculator LangGraph example: one full "agent turn, then tool turn" cycle is two graph steps, so the limit doubles the intended iteration budget.
- **`TavilySearch(topic="news")`** — Tavily supports a general topic too; `"news"` specifically biases toward dated news articles, which is why results here consistently carry publication dates.
- **The system prompt is unchanged from the other example** — proof that "which tool" and "how faithful the agent is to what it found" are independent concerns. The reliability guarantee (no fabrication) comes from the prompt and doesn't care which search backend is plugged in underneath.

## Extend it (optional exercises)

1. Swap `topic="news"` for the general topic and compare result quality/dates on the same company query.
2. Add a second tool (e.g. Tavily's `include_domains` restricted to a couple of financial sites) and see whether narrower, more targeted sources change what the summary reports.
3. Point this same graph at a different LangChain-compatible model (e.g. swap `ChatAnthropic` for another provider's chat model) and confirm only the model line changes — `TavilySearch` and the graph wiring don't care which LLM is driving them, unlike Claude's `web_search`, which is Anthropic-specific by definition.

## Where this fits your stock-analysis project

Same role as [`01_web_agent_langgraph`](../01_web_agent_langgraph/) — the qualitative/news-research half of a multi-agent system, as distinct from a data-fetcher agent using direct tool calls for exact numeric data (see [Stage 5](../../docs/stage-05-rag-retrieval.md)). Tavily specifically is worth knowing about here because a from-scratch multi-agent stock analysis system commonly wires up a dedicated news/search API exactly like this, independent of whichever LLM ends up driving the analysis.
