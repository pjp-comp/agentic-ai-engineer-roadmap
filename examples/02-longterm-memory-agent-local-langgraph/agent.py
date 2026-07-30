"""
A multi-turn chat agent that adds the third memory tier — long-term,
cross-session, durable facts — on top of examples/02-memory-agent-local-
langgraph. Same short-term window and persistent checkpointing as that
example, plus a fact store that survives regardless of window position.

The gap this fixes: in the short-term-only example, ask the agent to
"remember" something, keep chatting past WINDOW_SIZE turns (or across a
--reset), and it forgets — by design, because nothing outside the window
is visible to the model. Long-term memory here means: the model itself
decides a piece of information is worth keeping permanently, calls a
save_fact tool, and that fact gets injected into every future turn's
context (as a "known facts" block) regardless of how long the window is
or whether it's a brand-new conversation.

  LongTermMemory   A flat JSON file of facts (list of {key, value,
                    created_at}), separate from the turn-by-turn chat log.
                    remember() appends a fact (or overwrites one with the
                    same key — see "supersession" below). recall() returns
                    all current facts, formatted for the system prompt.

This is intentionally NOT a vector store — no embeddings, no similarity
search, no extra dependency beyond what examples/02-memory-agent-local-
langgraph already uses. It's the simplest thing that actually fixes "the
agent forgot something I told it to remember": an explicit, structured
key/value store the model writes to on purpose. See docs/stage-04-
memory-state.md's LongTermMemory class for the production version of this
idea (semantic recall over embedded facts, TTL/decay/supersession as
policy, not just overwrite-on-same-key).

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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "true").strip().lower() == "true"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.2:3b")
CLAUDE_MODEL = "claude-haiku-4-5"

WINDOW_SIZE = 20  # short-term memory holds the last 20 turns (10 exchanges)
STATE_FILE = Path(__file__).parent / ".chat_state.json"
FACTS_FILE = Path(__file__).parent / ".long_term_facts.json"
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Only answer using what's in the "
    "current conversation or in Known facts below — if something was never "
    "mentioned or saved, say you don't know it.\n\n"
    "Most messages need NO tool call at all — just answer directly. Only "
    "call save_fact when the user states a specific, durable fact about "
    "themselves or a plan (their name, a preference, a scheduled date) "
    "AND it is not already present in Known facts below.\n\n"
    "Examples of what TO save: \"my name is Alex\" -> save_fact(name, Alex). "
    "\"we start on the 30th\" -> save_fact(start_date, 30th).\n"
    "Examples of what NOT to save: questions (\"what should I learn "
    "first?\"), acknowledgements (\"ok thanks\"), commands (\"git status\"), "
    "answers you just gave (a topic recommendation, a calculation), or "
    "anything already listed in Known facts. If in doubt, do not call the "
    "tool — just answer the question.\n\n"
    "After a tool call, always follow up by directly answering what the "
    "user actually asked — never let a tool call replace your answer to "
    "their message."
)


# --- Tier 1: short-term memory (identical to 02-memory-agent-local-langgraph) --

class ShortTermMemory:
    def __init__(self, window: int = WINDOW_SIZE):
        self.buffer: deque[dict] = deque(maxlen=window)

    def add(self, role: str, content: str) -> None:
        self.buffer.append({"role": role, "content": content})

    def get(self) -> list[dict]:
        return list(self.buffer)

    def load(self, turns: list[dict]) -> None:
        self.buffer.extend(turns[-self.buffer.maxlen:])


# --- Tier 2: persistent state (identical to 02-memory-agent-local-langgraph) --

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


# --- Tier 3: long-term memory (new in this example) ------------------------
# A flat JSON file, keyed by a short label the model chooses (e.g. "name",
# "scheduled_start_date"). Saving a fact with a key that already exists
# overwrites the old value instead of appending a duplicate — this is a
# tiny version of the "supersession on write" policy from Stage 4's
# forgetting section: a new fact about the same thing replaces the old one
# rather than leaving both to confuse future recall.

class LongTermMemory:
    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def remember(self, key: str, value: str) -> None:
        facts = self._load()
        facts[key] = {"value": value, "updated_at": time.time()}
        self.path.write_text(json.dumps(facts, indent=2))

    def recall_all(self) -> dict:
        return self._load()

    def as_context_block(self) -> str:
        facts = self.recall_all()
        if not facts:
            return "(none saved yet)"
        return "\n".join(f"- {key}: {data['value']}" for key, data in facts.items())

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


_long_term_store: LongTermMemory | None = None  # bound at startup in run_chat()


@tool
def save_fact(key: str, value: str) -> str:
    """Save a fact permanently, so it's remembered in every future turn and
    every future conversation — not just until it scrolls out of the
    short-term window.

    key: a short label for what this fact is about (e.g. "name",
         "favorite_color", "scheduled_start_date").
    value: the fact itself, in a few words.
    """
    assert _long_term_store is not None, "long-term store not initialized"
    _long_term_store.remember(key, value)
    return f"Saved: {key} = {value}"


TOOLS = [save_fact]


# --- Model selection (identical pattern to 02-memory-agent-local-langgraph) --
#
# Two instances of the same underlying model: `llm` has save_fact bound so
# it can choose to call it, and `llm_text_only` does NOT — it's used only
# for the follow-up answer after a tool call. This matters specifically
# for small local models: Llama 3.2 3B, once it has tool_calls in its own
# recent history, tends to re-issue the same tool call again even when
# explicitly told to just answer in plain text — bind_tools isn't just a
# hint, it changes what the model is willing to output. Removing the tool
# binding entirely for that second pass makes a text reply the only
# possible output, instead of relying on prompting alone to redirect it.

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


def _system_prompt(long_term: LongTermMemory) -> str:
    return f"{SYSTEM_PROMPT}\n\nKnown facts:\n{long_term.as_context_block()}"


def _to_lc_messages(turns: list[dict], long_term: LongTermMemory) -> list:
    lc_messages = [SystemMessage(content=_system_prompt(long_term))]
    for turn in turns:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        lc_messages.append(cls(content=turn["content"]))
    return lc_messages


def run_chat() -> None:
    global _long_term_store

    llm, llm_text_only = _build_llm()
    persistent = PersistentState(STATE_FILE)
    short_term = ShortTermMemory()
    long_term = LongTermMemory(FACTS_FILE)
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

    existing_facts = long_term.recall_all()
    print(
        f"  [memory] {len(existing_facts)} long-term fact(s) loaded from {FACTS_FILE.name}",
        file=sys.stderr,
    )
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

        messages = _to_lc_messages(short_term.get(), long_term)
        response = llm.invoke(messages)

        # If the model asked to save a fact, run save_fact and give it a
        # second turn to respond in plain text — mirrors the ReAct loop from
        # examples/01-basic-agent, just bounded to at most one tool round
        # trip per user message since this agent only has one tool.
        #
        # The explicit HumanMessage nudge after the tool result matters: a
        # small model left to continue straight from its own tool_calls turn
        # tends to keep narrating the save ("I've saved that fact...")
        # instead of answering what the user actually asked. Re-pointing it
        # at the original question fixes that — without it, "ok thanks" or
        # "what should I learn first?" gets a save-confirmation instead of
        # a real reply.
        if response.tool_calls:
            messages.append(response)
            for call in response.tool_calls:
                print(f"  [tool call] {call['name']}({call['args']})", file=sys.stderr)
                result = save_fact.invoke(call["args"])
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
            messages.append(
                HumanMessage(
                    content=(
                        f"Now directly answer my original message: {user_input!r}"
                    )
                )
            )
            response = llm_text_only.invoke(messages)

        reply = response.content or "(no text response)"
        print(f"agent> {reply}\n")

        short_term.add("assistant", reply)
        all_turns.append({"role": "assistant", "content": reply})
        persistent.checkpoint(all_turns)


if __name__ == "__main__":
    if "--reset" in sys.argv:
        PersistentState(STATE_FILE).clear()
        print(f"Cleared {STATE_FILE.name} (long-term facts kept — use --forget to clear those)")
        sys.exit(0)
    if "--forget" in sys.argv:
        LongTermMemory(FACTS_FILE).clear()
        print(f"Cleared {FACTS_FILE.name}")
        sys.exit(0)
    if "--facts" in sys.argv:
        facts = LongTermMemory(FACTS_FILE).recall_all()
        if not facts:
            print("(no long-term facts saved yet)")
        else:
            for key, data in facts.items():
                print(f"{key}: {data['value']}")
        sys.exit(0)
    run_chat()
