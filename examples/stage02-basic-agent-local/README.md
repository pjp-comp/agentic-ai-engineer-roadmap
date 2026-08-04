[← Back to roadmap](../../README.md)

# Example 02 — The Basic Agent, Running Locally (Free, No API Key)

Identical graph to [`examples/stage03-tool-calling-langgraph/`](../stage03-tool-calling-langgraph/) — same tool, same nodes, same edges — but the model answering "what should I do next?" can be a free, open-weight model running entirely on your machine via [Ollama](https://ollama.com/), instead of a call to the Claude API. No API key, no per-token cost, works offline once the model is downloaded.

## The point of this example

Every `01-*` example calls the Claude API. But **the ReAct loop, the graph, and the tool-calling contract are not Claude-specific** — they're a property of the LangChain/LangGraph chat-model interface, which any tool-calling model can implement. This example proves that by swapping the model with a single `if` branch and changing nothing else. Starting with this example, every `02-*` example in this repo **defaults to the local model** (`USE_LOCAL_MODEL=true` by default, both in `agent.py`'s own fallback and in `.env.example`) — free, no API key, no billing risk while you're learning. Claude stays fully wired up as an explicit opt-in, not removed:

```python
if USE_LOCAL_MODEL:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=LOCAL_MODEL, temperature=0).bind_tools(TOOLS)
else:
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024).bind_tools(TOOLS)
```

Everything below that — `call_model`, the `StateGraph`, `ToolNode`, `tools_condition` — is copy-pasted from example stage03-tool-calling-langgraph, unchanged. That's the whole lesson: the model is a pluggable dependency, not the thing the graph is built around.

## Where the model actually lives

You are **not** downloading raw model weight files into this git repo. Ollama manages the model in its own local store (`~/.ollama/models` on macOS/Linux), the same way `uv`/`pip` manage packages in a venv rather than in your project folder. `USE_LOCAL_MODEL` and `LOCAL_MODEL` in your `.env` just tell this script which model *name* to ask Ollama for at runtime — the actual ~2GB of weights lives outside git, downloaded once by `ollama pull`, never committed.

## Setup — local model path (the default)

1. **Install Ollama**: [ollama.com/download](https://ollama.com/download) (macOS/Windows/Linux). This installs a background service that serves models over a local API at `http://127.0.0.1:11434`.
2. **Pull the model** (one-time, ~2GB download):
   ```bash
   ollama pull llama3.2:3b
   ```
3. **Copy `.env.example` to `.env`** at the repo root if you haven't already — `USE_LOCAL_MODEL=true` is already the default there, nothing to change:
   ```bash
   USE_LOCAL_MODEL=true
   LOCAL_MODEL=llama3.2:3b
   ```
4. **Run it**:
   ```bash
   cd examples/stage02-basic-agent-local
   uv run agent.py "What is 23 * 47, plus 100?"
   ```

Expected output:

```
  [model] local via Ollama: llama3.2:3b
  [tool call] calculate({'expression': '23 * 47 + 100'})
The result of 23 * 47 + 100 is 1181.
```

The first response is slower than Claude (CPU inference on a laptop, not a data center), but every run after that is still free and still works with no internet connection once the model is pulled.

## Setup — Claude API path (opt-in, set `USE_LOCAL_MODEL=false`)

```bash
# repo-root .env: ANTHROPIC_API_KEY=sk-ant-..., USE_LOCAL_MODEL=false
cd examples/stage02-basic-agent-local
uv run agent.py "What is 23 * 47, plus 100?"
```

## Why Llama 3.2 3B specifically

It's small enough (≈2GB quantized) to run on a laptop CPU at usable speed, and — unlike some small models — it has reliable native tool-calling support as of recent Ollama builds, which matters here: a model that can't emit well-formed tool calls will break `tools_condition`'s routing. If you swap in a different `LOCAL_MODEL`, verify it actually supports tool calling in Ollama before assuming this graph will work with it — not every small model does.

## What to look at closely

- **`llm = ChatOllama(...)` vs. `llm = ChatAnthropic(...)`** — both return an object satisfying the same LangChain chat-model interface (`.bind_tools()`, `.invoke()`, an `AIMessage` with `.tool_calls`). The graph never branches on which one it got.
- **`temperature=0` on the local model** — small open-weight models are more prone to inconsistent tool-call formatting at higher temperatures than a frontier model is; pinning to 0 makes this example reliable to run repeatedly. Feel free to raise it once you're comfortable with the failure modes.
- **No API key required for the local path** — this is genuinely useful for the earliest, highest-repetition part of learning: you can run this agent as many times as you want, with zero billing risk, while you're still getting comfortable with the ReAct loop itself.
- **Speed/quality tradeoff is real** — this is not "Claude for free." A 3B local model will misunderstand more, hallucinate more, and reason less reliably than Claude on genuinely hard tasks. Use the local default to learn the mechanics cheaply; set `USE_LOCAL_MODEL=false` for anything where output quality actually matters — that code path is still there, just not the default anymore.

## Where this goes next

[`examples/stage04-memory-agent/`](../stage04-memory-agent/) builds on this exact pattern — same local/Claude model switch — to add the short-term memory concept from [Stage 4](../../docs/stage-04-memory-state.md), so you can see memory management working end-to-end without spending API credits while you iterate on it.
