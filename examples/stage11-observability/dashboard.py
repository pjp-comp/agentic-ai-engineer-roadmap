"""
Stage 11's "build one dashboard view that surfaces p95 latency and $/run"
requirement, and separately its "Done when" bar: given a bad output, find
the exact span that caused it in under two minutes.

This reads .traces.jsonl (written by agent.py's Tracer) -- no external
dashboard tool, no hosted backend. It's the minimum real version of what
a LangSmith/Arize/Langfuse dashboard view shows, over the same span data.

Usage:
    uv run dashboard.py                    -> summary: p95 latency and
                                               $/run, per node and overall
    uv run dashboard.py --trace <trace_id> -> full span-by-span detail
                                               for one run, in order
"""

import sys
from statistics import quantiles

from tracing import load_spans


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    # quantiles() needs at least 2 data points; n=100 gives us the 95th directly
    return round(quantiles(values, n=100)[94], 1)


def summary() -> None:
    spans = load_spans()
    if not spans:
        print("No traces recorded yet. Run: uv run agent.py \"some topic\"")
        return

    traces = {}
    for s in spans:
        traces.setdefault(s["trace_id"], []).append(s)

    print(f"{len(traces)} trace(s), {len(spans)} span(s) total\n")

    # Per-node latency/cost, across every trace -- this is the "which
    # node is slow / expensive" view a real observability dashboard leads with.
    by_node: dict[str, list[dict]] = {}
    for s in spans:
        by_node.setdefault(s["name"], []).append(s)

    print(f"{'node':<12} {'calls':>6} {'p95 latency (ms)':>18} {'total tokens':>14} {'total $':>10}")
    for name, node_spans in sorted(by_node.items()):
        latencies = [(s["end"] - s["start"]) * 1000 for s in node_spans]
        tokens = sum(s["prompt_tokens"] + s["completion_tokens"] for s in node_spans)
        cost = sum(
            (s["prompt_tokens"] + s["completion_tokens"]) / 1000 * 0.0   # local models: always $0
            for s in node_spans
        )
        errors = sum(1 for s in node_spans if s.get("error"))
        err_note = f"  ({errors} error(s))" if errors else ""
        print(f"{name:<12} {len(node_spans):>6} {p95(latencies):>18} {tokens:>14} {cost:>10.4f}{err_note}")

    # Per-trace total latency and cost -- the "$/run" half of the brief.
    print(f"\n{'trace_id':<38} {'nodes':>6} {'total latency (ms)':>20} {'total $':>10}")
    for trace_id, trace_spans in traces.items():
        total_latency = sum((s["end"] - s["start"]) * 1000 for s in trace_spans)
        total_cost = sum(
            (s["prompt_tokens"] + s["completion_tokens"]) / 1000 * 0.0 for s in trace_spans
        )
        print(f"{trace_id:<38} {len(trace_spans):>6} {round(total_latency, 1):>20} {total_cost:>10.4f}")


def show_trace(trace_id: str) -> None:
    spans = [s for s in load_spans() if s["trace_id"] == trace_id]
    if not spans:
        print(f"No spans found for trace_id={trace_id}")
        return

    spans.sort(key=lambda s: s["start"])
    print(f"Trace {trace_id} -- {len(spans)} span(s), in order:\n")
    for s in spans:
        latency = round((s["end"] - s["start"]) * 1000, 1)
        status = f"ERROR: {s['error']}" if s.get("error") else "ok"
        print(f"[{s['name']}] {latency}ms  tokens={s['prompt_tokens'] + s['completion_tokens']}  status={status}")
        print(f"    input:  {s['input']}")
        print(f"    output: {s['output']}\n")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--trace":
        show_trace(argv[1])
    else:
        summary()
