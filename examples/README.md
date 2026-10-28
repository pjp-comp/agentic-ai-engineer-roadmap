[← Learning path](../README.md)

# Choose an example

Follow the root learning path first. **Core** means first-pass practice; **Comparison** repeats a task to teach an implementation difference; **Extension** and **Advanced** can wait. Keep the comparisons: deleting them would hide what each abstraction changes.

Commands below each example's own README are authoritative. Run from that directory. Python projects are isolated; shared concepts do not imply shared runtime state.

| Example | Role | Models / services to prepare | When to study it |
|---|---|---|---|
| [stage01-async-fanout](stage01-async-fanout/) | Core | None | Stage 01: timeouts and partial failures |
| [stage02-sampling-params](stage02-sampling-params/) | Core | qwen3:1.7b | Stage 02: sampling and token demonstrations |
| [stage02-model-router](stage02-model-router/) | Extension | llama3.2:3b + llama3.1:8b; optional Claude | After baseline evaluation: compare routing costs |
| [stage02-basic-agent-local](stage02-basic-agent-local/) | Comparison | llama3.2:3b; optional Claude | After Stage 03 LangGraph: compare model adapters |
| [stage03-tool-calling](stage03-tool-calling/) | Core | llama3.2:3b; optional Claude | First agent: raw SDK and arithmetic tool |
| [stage03-multi-tool-agent](stage03-multi-tool-agent/) | Extension | llama3.2:3b; optional Claude | After raw calculator: tool selection and dispatch |
| [stage03-tool-calling-langchain](stage03-tool-calling-langchain/) | Comparison | llama3.2:3b; optional Claude | After raw calculator: common model/tool interfaces |
| [stage03-tool-calling-langgraph](stage03-tool-calling-langgraph/) | Comparison | llama3.2:3b; optional Claude | After LangChain comparison: loop expressed as graph |
| [stage03-mcp-tool-server](stage03-mcp-tool-server/) | Extension | llama3.2:3b; optional Claude | After raw calculator: tool discovery over MCP |
| [stage03-web-agent-langgraph](stage03-web-agent-langgraph/) | Extension | Anthropic key + network; Claude only | After graph calculator: provider-hosted web search |
| [stage03-web-agent-tavily](stage03-web-agent-tavily/) | Extension | Tavily key + network + llama3.2:3b; optional Claude | After graph calculator: client-executed search |
| [stage04-memory-agent](stage04-memory-agent/) | Core | llama3.2:3b; optional Claude | After bounded loop: recent messages and saved chat |
| [stage04-longterm-memory-agent](stage04-longterm-memory-agent/) | Extension | llama3.2:3b; optional Claude | After memory agent: explicit facts in JSON |
| [stage04-longterm-memory-vectorstore](stage04-longterm-memory-vectorstore/) | Extension | llama3.2:3b + nomic-embed-text; optional Claude for chat | After JSON facts: semantic fact retrieval |
| [stage05-rag-langgraph](stage05-rag-langgraph/) | Extension | llama3.2:3b + nomic-embed-text; Ollama only | After tool calling and graph comparison: text RAG |
| [stage05-multimodal-rag-agent](stage05-multimodal-rag-agent/) | Advanced | llama3.1:8b + qwen2.5vl:7b + nomic-embed-text; own PDFs | After text RAG and sessions: PDF pipeline; see its README for extra downloads |
| [stage06-sessions-plain](stage06-sessions-plain/) | Core | llama3.2:3b; Ollama only | After memory: explicit session/state/event storage |
| [stage06-sessions-langgraph](stage06-sessions-langgraph/) | Comparison | llama3.2:3b; Ollama only | After plain sessions and graph comparison: SQLite checkpoints |
| [stage07-react-loop](stage07-react-loop/) | Core | llama3.2:3b; Ollama only | After raw calculator: bounded control and failures |
| [stage08-supervisor-langgraph](stage08-supervisor-langgraph/) | Extension | llama3.2:3b; Ollama only | After graph comparison, sessions, and single-agent loop |
| [stage09-human-in-the-loop](stage09-human-in-the-loop/) | Core | llama3.2:3b; Ollama only | After graph comparison and security: persisted approval |
| [stage10-eval-harness](stage10-eval-harness/) | Core | llama3.2:3b + qwen3:1.7b; Ollama only | After basic logging: fixed cases and fallible model judge |
| [stage11-observability](stage11-observability/) | Extension demo / core topic | llama3.2:3b; Ollama only | Bundled demo needs Stage 08; single-agent tracing can be learned first |
| [stage12-guardrails](stage12-guardrails/) | Core | llama3.2:3b; Ollama only | Before real side effects: permissions and untrusted content |
| [stage13-idempotent-tools](stage13-idempotent-tools/) | Core | None | After approval and evaluation: retry identity and cache behavior |
| [stage14-portfolio](stage14-portfolio/) | Checklist | No executable project | After one tested project: publish reproducible evidence |
| [news-event-storyline-rag](news-event-storyline-rag/) | Advanced design guide | No executable project | After retrieval and evaluation: model news events and storylines |

## Entry points and setup

Most projects run with `uv run agent.py` and any arguments in their README. Exceptions:

- Async service: `uv run uvicorn app:app --reload`; use its README's `curl` request in a second terminal.
- Sampling: `uv run sampling_params.py compare-temps` (other subcommands are documented there).
- RAG: run `uv run build_index.py` before asking questions. Read the multimodal README for PDF parsing and optional embedding dependencies.
- Tracing: run `agent.py` to write traces, then `uv run dashboard.py` to inspect them.
- Portfolio and news-storyline directories contain documentation, not runnable applications.

Pull each required local model with `ollama pull <tag>`. The sampling demo and evaluation judge use `qwen3:1.7b`, not the default calculator model. Multimodal processing may download additional embedding weights and require system packages; its first run can be substantially heavier than text RAG.

The local/Claude switch is implemented only where noted. Some scripts hard-code model tags even though the root `.env` defines `LOCAL_MODEL`. No-model examples need neither Ollama nor an API key. Web-search examples require internet access even with local inference. Initial dependency and model downloads also need network access.

Saved chats, facts, indexes, SQLite files, and traces are learning data. Keep them out of git and inspect reset/rebuild options before using them. Example output is illustrative and may vary across models, versions, and runs.
