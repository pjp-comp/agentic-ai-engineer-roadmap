[← Learning path](../README.md)

# Words used in this course

| Term | Plain-language meaning |
|---|---|
| LLM / model | A trained system that generates tokens from supplied context |
| Token | A tokenizer-defined piece of text or other input; not necessarily a word |
| Context window | The model's capacity for input and generated tokens, subject to model-specific limits |
| Prompt | Instructions and content sent to a model |
| Context engineering | Choosing which instructions, history, tool definitions, and evidence a call receives |
| Temperature | A sampling setting affecting variation; zero does not guarantee reproducibility |
| Inference | Running a trained model to generate output |
| Tool / function calling | The model requests an operation with structured arguments; application code validates and runs it |
| Schema | A description of expected fields and types; still validate values and permissions |
| Agent | A program in which a model selects some next actions toward a task |
| Workflow | Steps and routing primarily specified by application code |
| ReAct | A pattern that alternates model decisions, actions, and observations |
| Harness | The surrounding code for tools, state, limits, execution, and logging |
| LangChain | Model/tool abstractions used by several examples |
| LangGraph | A framework for expressing stateful execution as nodes and edges |
| State | Values describing the current task and its progress |
| Session / thread | An identifier and associated conversation or task history |
| Checkpoint | A stored snapshot used to resume execution; it does not automatically make external actions safe to repeat |
| Long-term memory | Selected information retained for use across sessions; it needs correction and deletion policies |
| Embedding | A numeric representation used to compare content by similarity |
| Vector store | Storage and search for embeddings and their associated content |
| RAG | Retrieval-augmented generation: retrieve external evidence, then use it to generate an answer |
| Chunk | A unit of source content selected for indexing or retrieval |
| Grounding | Basing a claim on available evidence; retrieval alone does not guarantee correctness |
| MCP | Model Context Protocol: a standard interface for discovering and accessing tools and other resources |
| Multi-agent | Multiple model-driven roles coordinated by application logic |
| Human in the loop | A person reviews or steers an action at a defined point |
| Prompt injection | Untrusted content attempts to redirect the model's instructions or actions |
| Allowlist | An explicit set of permitted operations, enforced by code |
| Sandbox | An enforced execution boundary restricting resources and capabilities; an ordinary subprocess is not one |
| Trace / span | A record of a whole run / one operation within it |
| Evaluation / rubric | A repeatable behavior check / the criteria used to judge it |
| Idempotency key | An identifier reused for retries of one logical action, distinct from a new action with identical inputs |
| TTL | Time to live: how long a cache entry or stored fact remains valid |
| CI / canary | Automated change checks / a release exposed to a small traffic share first |

Read definitions as working vocabulary, then connect each one to a concrete example. Frameworks may use the same word differently; check their API documentation.
