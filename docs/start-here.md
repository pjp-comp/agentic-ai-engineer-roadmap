[← Learning path](../README.md)

# Your first agent: a guided walkthrough

## Before you begin

You should be able to write a function, use lists and dictionaries, import a module, handle an exception with `try`/`except`, and run a Python file from a terminal. If these are new, work through the [Python tutorial](https://docs.python.org/3/tutorial/) first. Stage 01 then introduces `async`/`await`; you do not need machine-learning mathematics to run the first agent.

Keep a small learning log: **question → prediction → command → observed result → explanation**. Aim for one working example per study session. There is no deadline for completing the curriculum.

## 1. Prepare the local tools

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Ollama](https://ollama.com/download). Start the Ollama application or run `ollama serve` in a separate terminal if no server is already running.

From the repository root:

```bash
uv --version
ollama --version
ollama pull llama3.2:3b
ollama list
```

The download happens once per model. File size is not the same as RAM required; runtime memory and speed depend on quantization, context length, and hardware. Start with this small model and postpone the larger PDF examples.

## 2. Run one understandable task

```bash
cd examples/stage03-tool-calling
uv run agent.py "Calculate (23 * 47 + 100) / 3"
```

Expected arithmetic: `23 * 47 = 1081`, `1081 + 100 = 1181`, `1181 / 3 ≈ 393.6667`. The model might request one combined expression or several operations. Look for the tool-call log as well as the answer: a plausible final number alone does not prove the tool ran.

Now ask a question that needs no calculator:

```bash
uv run agent.py "Say hello in one sentence."
```

Predict: the model should answer directly. If it requests an unnecessary tool, record that as a behavior to evaluate later.

## 3. Read the loop, then change one thing

Open [agent.py](../examples/stage03-tool-calling/agent.py). Find these pieces in order:

1. `calculate`: Python performs the arithmetic.
2. Tool definitions: the name, description, and argument schema shown to the model.
3. The local-model loop: messages go to Ollama; requested tools are dispatched by Python.
4. Tool-result messages: observations return to the next model call.
5. `MAX_ITERATIONS`: the application stops repeated calls even if the model keeps asking.

The Claude branch implements the same idea with a different message format. You can skip that branch on your first reading. The local example also contains a fallback for a small model that writes a tool call as text; this is example-specific recovery logic, not a requirement of every tool API.

Change the tool description, rerun the same two questions, and compare tool choice. Restore it before proceeding. Then try `Calculate 1 / 0`: inspect the error path and whether the agent communicates the failure honestly. Exact recovery wording is model-dependent.

**Checkpoint:** explain who chooses the tool, who executes it, where the result goes, and who enforces the stopping limit. If you can explain all four, return to the [learning flow](../README.md#the-learning-flow) and work through its first-pass lessons.

## Configuration without losing existing settings

The calculator works locally with no `.env`. If you need configuration, return to the root and copy `.env.example` to `.env` only when `.env` is absent. Preserve any existing credentials and settings. Shell environment variables take precedence over dotenv values in the examples using default `load_dotenv` behavior.

`USE_LOCAL_MODEL=false` selects Claude only in examples that implement that switch; it does not convert an Ollama-only demo. Hosted calls may incur charges. The [catalog](../examples/README.md) lists the exceptions and extra models.

## When something goes wrong

| Symptom | Check | Next action |
|---|---|---|
| `uv: command not found` | Is uv installed and on PATH? | Follow its install guide; reopen the terminal |
| Connection refused on port 11434 | Is Ollama running? | Start the app or `ollama serve`; retry |
| Model not found | Does `ollama list` show the exact tag? | Pull the tag named in that example's README |
| Missing package/import | Are you in the example directory? | Run with `uv run`, using its own project and lockfile |
| API authentication error | Did you enable the hosted branch? | Use the local default or configure your own key |
| Slow or stalled generation | Is the model too large for available memory? | Start with the calculator model; close other heavy apps |
| Correct-looking text, no tool call | Did the model actually request a tool? | Inspect logs; try explicit arithmetic and record failures |
| Old memory appears | Are you reusing a session or saved store? | Read the example's reset options before clearing anything |
| Output differs from the README | Does the behavior still meet the check? | Compare tool use and correctness, rather than exact wording |

Do not delete saved conversations or indexes as a first troubleshooting step. Inspect which store/session is being used. Reset commands in individual guides remove local learning data and should be intentional.
