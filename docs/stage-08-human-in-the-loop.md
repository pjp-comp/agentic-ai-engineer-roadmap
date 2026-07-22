[← Stage 07](stage-07-multi-agent.md) · Stage 08 of 13 · **Next:** [Stage 09 →](stage-09-evaluation-qa.md)

# Stage 08 — Human-in-the-Loop Systems

Uncertainty detection · approval gates · audit trails · resume logic · intervention points

## Why this matters

Autonomous agents that touch money, production systems, or external comms need a pause button. The hard part isn't the UI — it's persisting enough state that a human can review, approve, edit, or reject mid-execution, and the agent resumes exactly where it left off.

## Brief

Add an interrupt before any tool call tagged `requires_approval=True`; persist graph state to durable storage at that point; expose a resume endpoint that replays from the checkpoint with the human's decision injected.

```python
graph = builder.compile(
    checkpointer=PostgresSaver(conn),
    interrupt_before=["execute_high_risk_action"],
)
# later, from an approval webhook:
graph.update_state(thread_id, {"approved": True})
graph.invoke(None, config={"configurable": {"thread_id": thread_id}})
```

## Sources

| Type | Resource |
|------|----------|
| Docs | [LangGraph — Human-in-the-loop / interrupts](https://docs.langchain.com/oss/python/langgraph/human-in-the-loop) |
| Docs | [LangSmith — run review & audit trail UI](https://docs.langchain.com/langsmith/observability) |
| Pattern | Saga / checkpoint pattern for resumable long-running workflows (Temporal docs are a good non-LLM reference) |

## Done when

You can kill the process mid-run at an approval gate, restart it, and it resumes without re-running completed side effects.

---
[← Stage 07 — Multi-Agent Orchestration](stage-07-multi-agent.md) · [Back to roadmap](../README.md) · **Next:** [Stage 09 — Evaluation + Quality Assurance →](stage-09-evaluation-qa.md)
