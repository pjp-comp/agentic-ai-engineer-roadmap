"""
A minimal, dependency-free span tracer -- the vendor-neutral half of
Stage 11's brief ("or use OpenTelemetry exporters for a vendor-neutral
pipeline"). No LangSmith account, no API key, nothing external: this
writes real OpenTelemetry-shaped spans to a local JSONL file, which is
enough to demonstrate every concept Stage 11 asks for (per-node spans,
input/output, token counts, latency, cost, a p95/$-per-run dashboard)
without requiring a hosted tracing backend for a learning example.

A Span here mirrors the fields a real OTel/LangSmith span would have:
  - trace_id   -- one per agent run (one graph.invoke() call)
  - span_id    -- one per node execution
  - parent_id  -- always the trace's root span, since this graph's nodes
                  don't nest further (a real multi-agent-of-multi-agent
                  system would nest spans here)
  - name       -- the node name ("researcher", "critic", "writer", ...)
  - start/end  -- wall-clock timestamps, from which latency_ms is derived
  - input/output -- truncated for readability, not omitted -- Stage 11's
                  whole point is being able to find WHAT a node saw and
                  produced, not just that it ran
  - tokens/cost -- Ollama is free and local, so cost is always $0 here,
                  but the field exists because a real deployment (Stage
                  13) would point this same pipeline at a paid API and
                  the dashboard code shouldn't need to change
"""

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

TRACE_LOG = Path(__file__).parent / ".traces.jsonl"

# Local models are free -- this constant exists purely so the SAME
# dashboard code works unmodified if LOCAL_MODEL were swapped for a paid
# API in a real deployment (Stage 13's point about serving config and
# cost being separately trackable).
COST_PER_1K_TOKENS = 0.0


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start: float
    end: float = 0.0
    input: str = ""
    output: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        return round((self.end - self.start) * 1000, 1)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return round(self.total_tokens / 1000 * COST_PER_1K_TOKENS, 6)


def _truncate(text: str, limit: int = 200) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


class Tracer:
    """One Tracer per process. start_trace() begins a new trace_id (one
    per graph.invoke() call); span() is a context manager wrapping a
    single node's execution.
    """

    def __init__(self, log_path: Path = TRACE_LOG):
        self.log_path = log_path
        self.trace_id: str | None = None

    def start_trace(self) -> str:
        self.trace_id = str(uuid.uuid4())
        return self.trace_id

    @contextmanager
    def span(self, name: str, input_data=None):
        if self.trace_id is None:
            self.start_trace()
        span = Span(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            parent_id=self.trace_id,   # flat graph -- every node is a direct child of the trace root
            name=name,
            start=time.time(),
            input=_truncate(input_data) if input_data is not None else "",
        )
        try:
            yield span
        except Exception as e:
            span.error = str(e)
            raise
        finally:
            span.end = time.time()
            span.output = _truncate(span.output)
            self._write(span)

    def _write(self, span: Span) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(span)) + "\n")


def load_spans(log_path: Path = TRACE_LOG) -> list[dict]:
    if not log_path.exists():
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]
