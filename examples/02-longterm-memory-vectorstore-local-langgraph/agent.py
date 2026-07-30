"""
The long-term memory agent from examples/02-longterm-memory-agent-local-
langgraph, but with the flat JSON fact store replaced by a real local
vector store (ChromaDB) — this is the "real semantic recall" upgrade that
example's README flags as the natural next step.

What changes vs. that example:

  Flat JSON version            This version
  --------------------         --------------------------------------------
  Every fact injected into     Only the top-K facts *relevant to the
  every prompt, unconditionally  current message* are retrieved and injected
  Exact key match only         Semantic similarity search (embeddings) —
                                recalls a fact even if the current question
                                doesn't share any words with how it was saved
  Cost grows with total facts  Cost grows with K (a fixed constant), no
  ever saved                   matter how many facts exist in the store

Same save_fact tool as before, same idea (the model decides what's worth
keeping permanently) — but recall() now asks "which saved facts are
relevant to what the user just asked?" via embedding similarity, instead
of "here is the entire list, every time."

Requires a local embedding model pulled via Ollama (separate from the
chat model — embeddings and chat completion are different jobs):
    ollama pull nomic-embed-text        (~274MB, one-time)

This is required even when USE_LOCAL_MODEL=false (Claude API for chat) —
Anthropic's API does not offer an embeddings endpoint, so this example
always embeds locally via Ollama regardless of which model answers you.
That's a real, deliberate asymmetry: the chat model is swappable, the
embedding model here is not (without further code changes).

Usage:
    uv run agent.py                 # interactive chat
    uv run agent.py --reset         # clear short-term/persistent chat log only
    uv run agent.py --forget        # clear long-term facts only
    uv run agent.py --facts         # print all saved long-term facts and exit
"""

import json
import os
import sys
import time
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "true").strip().lower() == "true"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.2:3b")
CLAUDE_MODEL = "claude-haiku-4-5"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

WINDOW_SIZE = 20  # short-term memory holds the last 20 turns (10 exchanges)
RECALL_K = 3  # how many long-term facts to retrieve per turn, at most
STATE_FILE = Path(__file__).parent / ".chat_state.json"
CHROMA_DIR = Path(__file__).parent / ".chroma_facts"
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful, concise assistant. Only answer using what's in the "
    "current conversation or in Relevant known facts below — if something "
    "was never mentioned or saved, say you don't know it.\n\n"
    "Most messages need NO tool call at all — just answer directly. Only "
    "call save_fact when the user states a specific, durable fact about "
    "themselves or a plan (their name, a preference, a scheduled date) "
    "AND it is not already present in Relevant known facts below.\n\n"
    "Examples of what TO save: \"my name is Alex\" -> save_fact(name, Alex). "
    "\"we start on the 30th\" -> save_fact(start_date, 30th).\n"
    "Examples of what NOT to save: questions, acknowledgements (\"ok "
    "thanks\"), commands, answers you just gave, or anything already "
    "listed below. If in doubt, do not call the tool — just answer.\n\n"
    "After a tool call, always follow up by directly answering what the "
    "user actually asked — never let a tool call replace your answer.\n\n"
    "Relevant known facts (retrieved for this message, may be empty):\n{facts}"
)


# --- Tier 1 + 2: identical to 02-memory-agent-local-langgraph --------------

class ShortTermMemory:
    def __init__(self, window: int = WINDOW_SIZE):
        self.buffer: deque[dict] = deque(maxlen=window)

    def add(self, role: str, content: str) -> None:
        self.buffer.append({"role": role, "content": content})

    def get(self) -> list[dict]:
        return list(self.buffer)

    def load(self, turns: list[dict]) -> None:
        self.buffer.extend(turns[-self.buffer.maxlen:])


class PersistentState:
    def __init__(self, path: Path):
        self.path = path

    def checkpoint(self, turns: list[dict]) -> None:
        self.path.write_text(json.dumps(turns, indent=2))

    def resume(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text())

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


# --- Tier 3: long-term memory, now vector-backed ----------------------------
# Same remember()/recall() shape as docs/stage-04-memory-state.md's
# LongTermMemory class. Each fact is stored as a Document whose page_content
# is a natural-language sentence (so it embeds meaningfully) and whose
# metadata carries the structured key/value + timestamp (so exact
# supersession — overwrite the same key — is still possible, same as the
# flat-JSON version).

class LongTermMemory:
    def __init__(self, persist_dir: Path, embeddings: OllamaEmbeddings):
        self.store = Chroma(
            collection_name="facts",
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    def remember(self, key: str, value: str) -> None:
        # Supersession on write: delete any existing fact with this key
        # before adding the new one, so recall() never returns two
        # conflicting values for the same key — same policy as the flat
        # JSON version, just expressed as a Chroma delete-then-add instead
        # of a dict overwrite.
        existing = self.store.get(where={"key": key})
        if existing["ids"]:
            self.store.delete(ids=existing["ids"])
        self.store.add_documents(
            [Document(
                page_content=f"{key}: {value}",
                metadata={"key": key, "value": value, "updated_at": time.time()},
            )],
            ids=[f"{key}-{time.time()}"],
        )

    def recall_relevant(self, query: str, k: int = RECALL_K) -> list[dict]:
        # Semantic search: retrieves facts related in *meaning* to the
        # query, not just facts whose key/value text literally overlaps
        # with it — this is what lets "what did we plan?" find a fact
        # saved as "schedule: 30th july" with no shared words at all.
        results = self.store.similarity_search(query, k=k)
        return [r.metadata for r in results]

    def recall_all(self) -> list[dict]:
        raw = self.store.get()
        return [
            {"key": m["key"], "value": m["value"], "updated_at": m["updated_at"]}
            for m in raw["metadatas"]
        ]

    def clear(self) -> None:
        all_ids = self.store.get()["ids"]
        if all_ids:
            self.store.delete(ids=all_ids)


_long_term_store: LongTermMemory | None = None  # bound at startup in run_chat()


@tool
def save_fact(key: str, value: str) -> str:
    """Save a fact permanently, so it can be recalled by meaning in any
    future turn or conversation — not just until it scrolls out of the
    short-term window, and not just on an exact keyword match.

    key: a short label for what this fact is about (e.g. "name",
         "favorite_color", "scheduled_start_date").
    value: the fact itself, in a few words.
    """
    assert _long_term_store is not None, "long-term store not initialized"
    _long_term_store.remember(key, value)
    return f"Saved: {key} = {value}"


TOOLS = [save_fact]


# --- Model selection ---------------------------------------------------------
# Two model instances (same pattern as 02-longterm-memory-agent-local-
# langgraph, see that file's comment for why): `llm` has save_fact bound,
# `llm_text_only` doesn't, used for the post-tool-call follow-up answer.

def _build_llm():
    if USE_LOCAL_MODEL:
        from langchain_ollama import ChatOllama

        print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)
        base = ChatOllama(model=LOCAL_MODEL, temperature=0.3)
    else:
        from langchain_anthropic import ChatAnthropic

        print(f"  [model] Claude API: {CLAUDE_MODEL}", file=sys.stderr)
        base = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024)
    return base.bind_tools(TOOLS), base


def _print_token_usage(response) -> None:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        print("  [tokens] not reported by this model", file=sys.stderr)
        return
    print(
        f"  [tokens] in={usage['input_tokens']} "
        f"out={usage['output_tokens']} "
        f"total={usage['total_tokens']}",
        file=sys.stderr,
    )


def _facts_block(facts: list[dict]) -> str:
    if not facts:
        return "(none relevant)"
    return "\n".join(f"- {f['key']}: {f['value']}" for f in facts)


def _to_lc_messages(turns: list[dict], relevant_facts: list[dict]) -> list:
    system = SYSTEM_PROMPT_TEMPLATE.format(facts=_facts_block(relevant_facts))
    lc_messages = [SystemMessage(content=system)]
    for turn in turns:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        lc_messages.append(cls(content=turn["content"]))
    return lc_messages


def run_chat() -> None:
    global _long_term_store

    llm, llm_text_only = _build_llm()

    print(f"  [embeddings] local via Ollama: {EMBEDDING_MODEL}", file=sys.stderr)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    persistent = PersistentState(STATE_FILE)
    short_term = ShortTermMemory()
    long_term = LongTermMemory(CHROMA_DIR, embeddings)
    _long_term_store = long_term

    saved_turns = persistent.resume()
    if saved_turns:
        short_term.load(saved_turns)
        print(
            f"  [memory] resumed {len(short_term.get())} turn(s) from {STATE_FILE.name}",
            file=sys.stderr,
        )
    else:
        print("  [memory] starting fresh — no saved conversation found", file=sys.stderr)

    fact_count = len(long_term.recall_all())
    print(f"  [memory] {fact_count} long-term fact(s) in the vector store", file=sys.stderr)
    print(f"  [memory] short-term window holds the last {WINDOW_SIZE} turns\n", file=sys.stderr)
    print("Type a message, or 'quit' to exit.\n")

    all_turns = saved_turns.copy()

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in {"quit", "exit"}:
            break

        short_term.add("user", user_input)
        all_turns.append({"role": "user", "content": user_input})

        relevant_facts = long_term.recall_relevant(user_input)
        if relevant_facts:
            keys = ", ".join(f["key"] for f in relevant_facts)
            print(f"  [recall] retrieved facts relevant to this message: {keys}", file=sys.stderr)

        messages = _to_lc_messages(short_term.get(), relevant_facts)
        response = llm.invoke(messages)

        if response.tool_calls:
            messages.append(response)
            for call in response.tool_calls:
                print(f"  [tool call] {call['name']}({call['args']})", file=sys.stderr)
                result = save_fact.invoke(call["args"])
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
            # Tools unbound for the follow-up — see the comment on
            # _build_llm(): a small local model tends to re-issue the same
            # tool call again instead of answering if tools are still bound.
            response = llm_text_only.invoke(messages)

        reply = response.content or "(no text response)"
        print(f"agent> {reply}")
        _print_token_usage(response)
        print()

        short_term.add("assistant", reply)
        all_turns.append({"role": "assistant", "content": reply})
        persistent.checkpoint(all_turns)


if __name__ == "__main__":
    if "--reset" in sys.argv:
        PersistentState(STATE_FILE).clear()
        print(f"Cleared {STATE_FILE.name} (long-term facts kept — use --forget to clear those)")
        sys.exit(0)
    if "--forget" in sys.argv:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        LongTermMemory(CHROMA_DIR, embeddings).clear()
        print("Cleared all long-term facts from the vector store")
        sys.exit(0)
    if "--facts" in sys.argv:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        facts = LongTermMemory(CHROMA_DIR, embeddings).recall_all()
        if not facts:
            print("(no long-term facts saved yet)")
        else:
            for f in facts:
                print(f"{f['key']}: {f['value']}")
        sys.exit(0)
    run_chat()
