# Learn Agentic AI, One Working Program at a Time

Build an agent you can explain, test, and improve. This repository contains **14 topic guides and 25 independent Python examples**, plus two documentation-only project guides. You do not need to learn every framework or run every example.

**Start with [the beginner walkthrough](docs/start-here.md).** It covers prerequisites, setup, your first agent, and common errors. Use the [example catalog](examples/README.md) to choose a runnable project and the [glossary](docs/glossary.md) when a term is unfamiliar.

## What you are learning

An **LLM** generates a response from the context you give it. A **workflow** follows steps chosen by your code. An **agent** lets the model choose some next steps, such as which tool to request, while your application validates and executes those requests.

**Agentic AI** is a broad term for systems with this goal-directed behavior. A single agent can be agentic; multiple agents, persistent memory, and long-running autonomy are optional design choices. Terminology varies, so describe the actual control flow rather than treating these words as certification levels. See Anthropic's [workflow/agent distinction](https://www.anthropic.com/engineering/building-effective-agents).

A common tool-using loop is:

```text
user task → model → final answer
               ↓ tool request
         validate + authorize
               ↓
         execute tool → return result to model
```

Your code owns permissions, timeouts, and stopping limits. A tool request is data to validate, not permission to act. ReAct (reasoning and acting) is one useful pattern; it does not require printing the model's private reasoning. Use an ordinary function or fixed workflow when its steps already solve the problem.

## The learning flow

Stage numbers identify topics and keep existing links stable. **Follow the order below for your first pass.** Examples are separate teaching programs, not one application that automatically accumulates every stage's features.

| Step | Topic guide | First task | Evidence you are ready to continue |
|---|---|---|---|
| 1 | [01: Python and async](docs/stage-01-python-async.md) | Run concurrent mock tools | One failure leaves the other results available |
| 2 | [02: LLM fundamentals](docs/stage-02-llm-fundamentals.md) | Try sampling; postpone model routing | Explain tokens, context, and variable outputs |
| 3 | [03: Tool calling](docs/stage-03-tool-calling.md) | Run the raw-SDK calculator | Trace a request, tool call, result, and answer |
| 4 | [07: Single-agent control](docs/stage-07-single-agent.md) | Add bounded retries and stopping rules | An impossible request terminates clearly |
| 5 | [04: Memory](docs/stage-04-memory-state.md) | Try a bounded conversation buffer | Explain what was forgotten and why |
| 6 | [06: Sessions](docs/stage-06-sessions-state.md) | Save and reopen one session | A second process resumes the correct session |
| 7 | [12: Security](docs/stage-12-security-guardrails.md) | Enforce tool permissions in Python | A forbidden tool is blocked before execution |
| 8 | [09: Human approval](docs/stage-09-human-in-the-loop.md) | Pause before a simulated refund | Rejection produces no refund; approval resumes |
| 9 | [11: Observability](docs/stage-11-observability.md) | Record model/tool steps | Locate the step behind a bad answer |
| 10 | [10: Evaluation](docs/stage-10-evaluation-qa.md) | Keep a small fixed case set | A deliberate regression fails a check |
| 11 | [13: Deployment](docs/stage-13-deployment.md) | Run the no-model retry demo | Distinguish a retry from a new action |
| 12 | [14: Portfolio](docs/stage-14-portfolio.md) | Document one finished project | Another person can run and assess it |

Start logging requests and checking a few known answers at Step 3; Steps 9–10 deepen those habits. Read security and approval guidance **before connecting any real writing, payment, sending, or deletion tool**. Early examples should use arithmetic and simulated data.

Two implementation choices in the route need care: Stage 09 uses LangGraph, so first complete Stage 03's optional LangGraph calculator comparison. Stage 11's bundled demo instruments Stage 08's supervisor; on your first pass, apply the tracing ideas to your single agent and defer that demo. Stage 10's bundled harness evaluates its own small agent; adapt its checks to your project yourself.

## Extensions after the first agent

Choose an extension because your project needs it:

- **Frameworks:** compare raw SDK → LangChain components → LangGraph in [Stage 03](docs/stage-03-tool-calling.md). These intentionally repeat the calculator task so the implementation differences are visible.
- **Document questions:** [Stage 05: RAG](docs/stage-05-rag-retrieval.md), after Stage 03. Build text retrieval before attempting PDFs, tables, or images.
- **Remembering across sessions:** Stage 04's JSON facts, then vector retrieval. These are optional; session persistence alone does not require a vector database.
- **Delegation:** [Stage 08: multi-agent orchestration](docs/stage-08-multi-agent.md), after Stages 03, 06, and 07 plus the LangGraph comparison. Measure it against a single-agent baseline.
- **External tools:** MCP and web-search examples in Stage 03, after the local calculator. Web search requires network access and provider credentials as listed in the catalog.
- **Advanced projects:** multimodal RAG and the news-storyline design guide, after basic retrieval and evaluation.

## Run an example

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Ollama](https://ollama.com/download), and start Ollama. Run from the repository root:

```bash
uv --version
ollama --version
ollama pull llama3.2:3b
cd examples/stage03-tool-calling
uv run agent.py "Calculate (23 * 47 + 100) / 3"
```

Expect a calculator tool call and a result near **393.6667**. Wording and call count vary. The first `uv run` installs dependencies and may download Python; local model inference is slower on some machines. The examples declare Python 3.10 or newer; a dependency may impose a stricter version.

No `.env` is required for this local default. To configure a supported example, copy `.env.example` to `.env` at the root **only if `.env` does not already exist**, then edit it. Keep credentials in `.env`, never in `.env.example` or a commit. Some examples support Claude; others are Ollama-only. `USE_LOCAL_MODEL` and `LOCAL_MODEL` do not affect scripts that hard-code their model. Consult the [catalog](examples/README.md) before downloading additional models.

Each runnable example has its own `pyproject.toml` and `uv.lock`. Run commands inside that example's directory. Once dependencies and models are downloaded, examples using only local data can run offline. Local inference has no hosted token bill, but uses your hardware and electricity; web-search examples remain online.

## How to study each lesson

1. Read its **First pass** section and predict the output before running it.
2. Run the linked example and find the function or graph edge responsible for each step.
3. Make one small change; record what you expected and what happened.
4. Trigger one failure and check how the program stops or recovers.
5. Explain the result without reading the code. Continue when the completion check is observable.

Code blocks in topic guides are **design sketches unless explicitly labeled runnable**. Placeholder methods, services, and CI commands need implementation. Example READMEs provide actual commands. Sample outputs and historical measurements illustrate behavior; they are not promises that a model will reproduce them.

## Repository map

| Location | Purpose |
|---|---|
| [docs/start-here.md](docs/start-here.md) | Prerequisites, first run, and troubleshooting |
| [docs/glossary.md](docs/glossary.md) | Plain-language definitions |
| `docs/stage-*.md` | Topic lessons, first-pass exercises, and advanced reading |
| [examples/README.md](examples/README.md) | All examples, prerequisites, and model requirements |
| `examples/<name>/README.md` | Commands and code-reading notes for that example |
| [.env.example](.env.example) | Optional configuration template |
| [docs/repository-review.md](docs/repository-review.md) | Review scope and remaining teaching-demo limits |

Prefer linked official documentation for changing APIs. Use the locked dependencies to reproduce an example; evaluate and test upgrades separately.

## License

No license file is included. Do not assume this repository grants redistribution rights; add an explicit license if you own the material and intend to publish it for reuse.
