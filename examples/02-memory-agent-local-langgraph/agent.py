"""
A multi-turn chat agent that demonstrates the first two memory tiers from
Stage 4 (docs/stage-04-memory-state.md) — short-term and persistent state
— running against a free local Ollama model or the Claude API, same
USE_LOCAL_MODEL switch as examples/02-basic-agent-local-langgraph.

What this adds on top of that example:

  ShortTermMemory   A bounded deque(maxlen=WINDOW_SIZE) of the last N turns.
                    Old turns are dropped, not summarized — this is the
                    simplest tier, and dropping is the point: you can watch
                    the agent forget something once it ages out of the window.

  PersistentState   The whole conversation checkpointed to a JSON file after
                    every turn, so killing this process (Ctrl+C) and running
                    it again resumes exactly where you left off — the window
                    is rebuilt from the last WINDOW_SIZE turns in the file,
                    not lost on restart.

Long-term memory from Stage 4 is intentionally NOT included here — this
example only covers the first two tiers. That means facts WILL still be
forgotten once a conversation runs past WINDOW_SIZE turns, by design: ask
something in turn 1, keep chatting past turn 20, and asking about it again
gets an honest "I don't know" — that's the short-term tier's tradeoff, not
a bug. If you want a fact to survive indefinitely regardless of window
size, that's what the long-term tier is for — see
examples/02-longterm-memory-agent-local-langgraph, which builds on this
exact file and adds a save_fact tool + durable JSON fact store on top.

Usage:
    uv run agent.py                 # interactive chat
    uv run agent.py --reset         # clear saved state, start fresh
"""

import json
import os
import sys
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").strip().lower() == "true"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.2:3b")
CLAUDE_MODEL = "claude-haiku-4-5"

WINDOW_SIZE = 20  # short-term memory holds the last 20 turns (10 exchanges)
STATE_FILE = Path(__file__).parent / ".chat_state.json"
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Only answer using what's in the "
    "current conversation — if something was never mentioned, say you don't know it."
)


# --- Tier 1: short-term memory (in-process, bounded, exact recall) --------
# Same shape as ShortTermMemory in docs/stage-04-memory-state.md — a plain
# deque with a max length. Once it's full, appending a new turn silently
# drops the oldest one. Nothing is summarized; what's in the window is
# recalled verbatim, and what's outside it is gone from the model's view.

class ShortTermMemory:
    def __init__(self, window: int = WINDOW_SIZE):
        self.buffer: deque[dict] = deque(maxlen=window)

    def add(self, role: str, content: str) -> None:
        self.buffer.append({"role": role, "content": content})

    def get(self) -> list[dict]:
        return list(self.buffer)

    def load(self, turns: list[dict]) -> None:
        # Only the *last* window-worth of turns matters on resume — anything
        # older than that was already outside the window before we exited.
        self.buffer.extend(turns[-self.buffer.maxlen:])


# --- Tier 2: persistent state (survives a process restart) ----------------
# Same idea as PersistentState in docs/stage-04-memory-state.md, but backed
# by a JSON file instead of Redis/Postgres — same interface (checkpoint/
# resume), simplest possible store, so this example needs no extra service
# running beyond Ollama itself.

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


# --- Model selection (identical pattern to example 02-basic-agent-local-langgraph) --
# Built lazily (only when actually chatting, not on --reset) so clearing
# saved state never requires Ollama or an API key to be configured.

def _build_llm():
    if USE_LOCAL_MODEL:
        from langchain_ollama import ChatOllama

        print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)
        return ChatOllama(model=LOCAL_MODEL, temperature=0.3)
    else:
        from langchain_anthropic import ChatAnthropic

        print(f"  [model] Claude API: {CLAUDE_MODEL}", file=sys.stderr)
        return ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024)


def _to_lc_messages(turns: list[dict]) -> list:
    lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in turns:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        lc_messages.append(cls(content=turn["content"]))
    return lc_messages


def run_chat() -> None:
    llm = _build_llm()
    persistent = PersistentState(STATE_FILE)
    short_term = ShortTermMemory()

    saved_turns = persistent.resume()
    if saved_turns:
        short_term.load(saved_turns)
        print(
            f"  [memory] resumed {len(short_term.get())} turn(s) from {STATE_FILE.name}",
            file=sys.stderr,
        )
    else:
        print("  [memory] starting fresh — no saved state found", file=sys.stderr)

    print(f"  [memory] short-term window holds the last {WINDOW_SIZE} turns\n", file=sys.stderr)
    print("Type a message, or 'quit' to exit.\n")

    all_turns = saved_turns.copy()  # full history persisted to disk (unbounded)

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

        response = llm.invoke(_to_lc_messages(short_term.get()))
        reply = response.content or "(no text response)"
        print(f"agent> {reply}\n")

        short_term.add("assistant", reply)
        all_turns.append({"role": "assistant", "content": reply})

        # Checkpoint after every turn — same reasoning as Stage 4's
        # PersistentState: a crash mid-conversation should lose at most
        # the in-flight turn, not the whole history.
        persistent.checkpoint(all_turns)


if __name__ == "__main__":
    if "--reset" in sys.argv:
        PersistentState(STATE_FILE).clear()
        print(f"Cleared {STATE_FILE.name}")
        sys.exit(0)
    run_chat()
