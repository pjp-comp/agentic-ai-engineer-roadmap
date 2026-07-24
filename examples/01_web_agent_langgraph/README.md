[← Back to roadmap](../../README.md)

# Example 01d — A Web Research Agent (LangGraph + Claude's `web_search`)

Give it a company name, get back a summary of its recent news — grouped by category, dated, and built only from what the agent actually found. This is the first example in this repo that does real, agentic work instead of arithmetic: fetch → read → analyze → summarize.

```bash
uv run agent.py "polycab india limited" --days 7
```

## What's different about this one

Every earlier example (01-basic-agent, 01-basic-agent-langchain, 01-basic-agent-langgraph) uses a **client-side** tool — a Python function *you* write and run, whether that's a hand-rolled dispatch loop or LangGraph's `ToolNode`. This example uses `web_search`, one of Claude's **server-side** tools: Anthropic's own infrastructure runs the actual search and hands the results back to the model inline, in the same API response. There is no local function called `web_search()` anywhere in `agent.py` — there's nothing to execute, because Claude never asks *you* to run it.

That changes the graph shape. Compare to [`01-basic-agent-langgraph`](../01-basic-agent-langgraph/):

| | Calculator (LangGraph) | This example |
|---|---|---|
| Tool location | Client-side (your Python function) | Server-side (Anthropic's infrastructure) |
| Graph nodes | `agent` + `tools` (`ToolNode`) | `agent` only |
| Who runs the tool | `ToolNode`, in your process | Anthropic, inside the same API call |
| Why the loop exists | You need to run the tool, then call the model again with the result | You don't — the model already has the result before it replies |

The agent can still call `web_search` multiple times per turn (you'll see several `[web_search] ...` lines on stderr for one run) — it's just that every one of those calls, and the model's reaction to each result, happens inside a single `llm.invoke()` on Anthropic's side. LangGraph's job here is smaller than in the calculator example: one node, no conditional routing, because there's no local tool-execution step to route to.

## What it is

- **One tool**: `web_search`, Claude's built-in server-side search — declared as `{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}`. `max_uses` is the equivalent of `MAX_ITERATIONS` in the other examples: a hard cap so one turn can't spiral into unlimited searches.
- **A system prompt with real rules**: only summarize what was actually found (no fabricated headlines), prefer dated results, group by category (earnings / regulatory / market commentary), and stay descriptive — not investment advice.
- **A one-node graph**: `START → agent → END`. No `ToolNode`, because there's no client-side tool to dispatch to.

## Run it

```bash
cd examples/01_web_agent_langgraph
uv run agent.py "polycab india limited" --days 7
```

Uses the same repo-root `.env` as the other examples (see [`.env.example`](../../.env.example)) via `python-dotenv` — no `export` needed.

`--days` controls the requested news window (default 7). The company name is whatever you pass as the remaining arguments.

Expected output (search queries print to stderr so stdout stays clean prose):

```
  [web_search] Polycab India news July 2026
  [web_search] Polycab India Limited latest news
  [web_search] Polycab India results announcement 2026
  [web_search] Polycab India Q1 FY27 results July 16 2026 revenue PAT
  [web_search] Polycab India stock July 21 2026 analyst brokerage
```

```
### Q1 FY27 Results (announced July 16, 2026)
Polycab India reported record quarterly earnings and sales growth for Q1 FY27,
though commentary noted the year-over-year trend shifting from strongly positive
to flatter growth.

### Investor Relations / Corporate
- USA non-deal roadshow (July 20–27, 2026): organized by Jefferies, disclosed
  under Regulation 30 of SEBI (LODR) Regulations, 2015.

### Market / Share-price Commentary
- Stock closed at ₹8,885.40 on July 17, 2026, down 3.59% from the previous close.
- Over the past year: +29.22%, vs. Sensex -5.35% over the same period.
- Derivatives activity: notable open-interest increase noted around mid-July.

Caveats: the Q1 FY27 result was formally announced one day before the requested
window; most reporting on it lands on/after the window start. This is
informational summarization only, not investment advice.
```

Every line above traces back to an actual `web_search` result — nothing is invented, and the agent says so explicitly when a query window is thin.

**If the search tool is rate-limited** (Claude's `web_search` has its own usage limits, separate from your regular API rate limits), the agent won't fabricate a summary to cover for it — it says so plainly and suggests retrying:

```
I was unable to retrieve any news about Polycab India Limited. The web search
tool was rate-limited/unavailable throughout this attempt, so I have no search
results to summarize. ... I'd rather tell you plainly than fabricate anything.
```

That refusal-to-fabricate behavior is the system prompt's "only summarize news you actually found" rule doing its job — worth testing deliberately (see Extend it, below).

## What to look at closely

- **No `ToolNode` in this file** — grep for it if you don't believe it. The entire tool-execution step that the calculator examples need is absent, because Claude's servers execute `web_search` themselves.
- **`server_tool_use` and `web_search_tool_result` content blocks** — these are the block types that show up in the response when a server-side tool runs. Compare to `tool_use` / `tool_result`, the client-side equivalents from the calculator examples. `research_company()` walks these to print which queries actually ran and to catch a `max_uses_exceeded` or other search error, since those come back as a normal successful response with an error object inside `.content` — not a raised exception.
- **`final.content` is usually a list, not a string** — once citations or multiple reasoning segments are involved, the response's content is a list of `{"type": "text", "text": ...}` blocks rather than one string. `research_company()` joins just the text blocks; printing `.content` directly (an earlier version of this file did) dumps a raw Python list.
- **The system prompt is doing real work here** — "only summarize news you actually found" and "say so plainly" are what make the rate-limited example above refuse to hallucinate instead of confidently inventing headlines. This is the corrective-RAG mindset from [Stage 5](../../docs/stage-05-rag-retrieval.md) applied to search results instead of a vector store.

## Extend it (optional exercises)

1. Set `max_uses` down to `1` and ask about a company with genuinely developing news — watch the agent explicitly say it couldn't gather enough to be thorough, rather than answering confidently from one search.
2. Add `allowed_domains` or `blocked_domains` to `WEB_SEARCH_TOOL` (e.g. restrict to a couple of financial news sites) and compare result quality/consistency against the unrestricted default.
3. Force the rate-limit path on purpose (run several queries back-to-back) and confirm the "I'd rather tell you plainly than fabricate anything" behavior holds — this is the same reliability property Stage 5's corrective RAG example is built around, just triggered by a different kind of empty/unusable retrieval.

## Where this fits your stock-analysis project

This is the shape of the *news/sentiment* half of a multi-agent stock analysis system: a research agent that pulls qualitative, unstructured information (news, filings, commentary) and summarizes it — as distinct from a *data-fetcher* agent that would use direct tool calls against a market-data API for exact numbers (prices, ratios). Don't ask this agent for a stock price — see [Stage 5's "retrieval vs. tool calls" section](../../docs/stage-05-rag-retrieval.md) for why that split matters.
