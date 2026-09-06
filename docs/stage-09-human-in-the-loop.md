[← Stage 08](stage-08-multi-agent.md) · Stage 09 of 14 · **Next:** [Stage 10 →](stage-10-evaluation-qa.md)

# Stage 09 — Human-in-the-Loop Systems

Uncertainty detection · the four intervention patterns · approval gates · deciding what needs a human · audit trails · resume logic · the operational cost of asking

## Why this matters

Autonomous agents that touch money, production systems, or external comms need a pause button. The hard part isn't the UI — it's persisting enough state that a human can review, approve, edit, or reject mid-execution, and the agent resumes exactly where it left off.

There's a second, less obvious reason this stage matters: **an approval gate is the only guardrail that works against a failure you didn't anticipate.** [Stage 12's](stage-12-security-guardrails.md) defenses all require you to have predicted the attack — you write a pattern, you write an allowlist. A human looking at a proposed action catches the case nobody wrote a rule for. That makes this stage the backstop for every other safety mechanism in the roadmap, not merely a compliance checkbox.

## The four intervention patterns

"Human-in-the-loop" gets used for four genuinely different mechanisms with different costs and different failure modes. Knowing which one a situation calls for is most of the design work:

| Pattern | The human's role | Cost | Reach for it when |
|---|---|---|---|
| **Approve / reject** | Gate a proposed action before it runs — this stage's brief | Blocks the run until answered | The action has real, hard-to-reverse side effects: money, external comms, deletion, production changes |
| **Edit** | Modify the agent's proposal, then let it proceed | Blocks, and needs a UI that can render *and* accept a structured edit | The agent is usually 90% right and a human correction is faster than a re-prompt — drafted emails, generated configs |
| **Escalate / take over** | The agent gives up and hands the whole task to a person | Blocks, and the human needs full context to continue | The agent has detected it can't proceed — low confidence, repeated failure, an out-of-scope request |
| **Review after the fact** | The action already ran; a human audits it later | Doesn't block | The action is cheap to reverse and blocking would destroy the product's value. Note this is an **audit trail**, not a gate — it catches patterns, never the individual bad action |

The mistake worth naming: **treating review-after-the-fact as if it were a gate.** Logging an action for later review does not prevent it. If the action can't be undone, only a blocking pattern is a control.

## Deciding what needs a human — the actual hard part

Gating everything makes the agent useless; gating nothing makes it dangerous. Three criteria, applied in order:

1. **Reversibility.** Can this be undone cheaply? Sending an email, charging a card, deleting a record, posting publicly — all irreversible, all gate candidates. Reading data, drafting text, running a query — reversible, gate none of them. *This is the primary axis, and it's a property of the action, not of the model's confidence.*
2. **Blast radius.** Does it affect one record or ten thousand? A bulk operation deserves a gate even when the same action on a single record wouldn't.
3. **Confidence.** *Only then* consider whether the agent is unsure. An uncertain agent doing something reversible doesn't need a human; a confident agent wiring money does.

**Confidence is the criterion people reach for first, and it's the weakest of the three.** A model's stated confidence is not calibrated — it will confidently propose the wrong refund amount. Use confidence to trigger *escalation* (the agent knows it's stuck), not to decide whether a dangerous action is safe. Base that on reversibility.

**Uncertainty detection, done well.** Where confidence *is* useful, don't ask the model "how confident are you?" — that self-report is nearly worthless. Better signals: the loop hit its stagnation check ([Stage 7](stage-07-single-agent.md)), retrieval graded weak ([Stage 5's CRAG refusal](stage-05-rag-retrieval.md)), a tool errored repeatedly, or a value falls outside a known-safe range (a refund above the largest one ever legitimately issued). All four are observable facts about the run rather than the model's opinion of itself — and a threshold on a proposed *value* is often the single most effective gate you can write.

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

**Runnable version:** [`examples/stage09-human-in-the-loop/`](../examples/stage09-human-in-the-loop/) builds this exact interrupt/resume flow — `interrupt_before` on a high-risk action node, a `SqliteSaver` checkpoint (standing in for `PostgresSaver` above), and an approve/reject decision applied in a genuinely separate process, proving the resume isn't just "the variable stayed in scope." Runs entirely on Ollama.

## Why the interrupt goes *before* the action, not after

The brief's `interrupt_before=["execute_high_risk_action"]` looks like a detail; it's the whole design. Pausing after the node runs gives you an audit log of something that already happened. Pausing before it gives you a gate. This is why the split matters in the graph's shape: **the node that decides what to do and the node that does it must be separate nodes**, because you can only interrupt at a node boundary.

That's a genuinely useful structural rule beyond LangGraph: keep "draft the action" and "perform the action" as separate steps in *any* framework, even without an approval gate today. Merging them means adding a gate later requires restructuring rather than adding a config line.

## Audit trails — what to record, and why "who" is the hard part

An approval gate that doesn't record its decisions is a control you can't prove you had. For anything touching money, health, or personal data, the trail *is* the compliance artifact. Record, per decision:

| Field | Why |
|---|---|
| The **proposed** action, in full | What was actually put in front of the human — not what eventually ran |
| The agent's **reasoning** for proposing it | Makes a pattern of bad proposals visible before someone approves one |
| **Who** decided, and **when** | The part that's legally load-bearing, and the part most implementations skip |
| The **decision** and any rejection reason | Rejection reasons are training data for the next prompt revision |
| The action **as executed**, and its result | An approved action can still fail; approval isn't confirmation |

Two properties that separate an audit trail from a log: it must be **append-only** (a mutable audit trail proves nothing) and it must record **the proposal separately from the execution** (approving `$45` and executing `$450` is exactly the failure an audit trail exists to catch — so store both and reconcile them).

Note the overlap with [Stage 11's tracing](stage-11-observability.md): the trace records what happened, the audit trail records who authorized it. Same events, different retention and different integrity requirements — traces get sampled and expire in weeks, audit records get kept whole for years.

## What "resume" actually requires

The brief's `graph.invoke(None, config=...)` hides three requirements that are easy to get wrong:

1. **Durable, external checkpoint storage.** If state lives in process memory, "resume" means "as long as nobody restarts anything," which is not a feature. The runnable example uses SQLite precisely so a decision made in a *separate process* proves the point.
2. **A stable thread identifier the human's decision can reference.** This ID travels into an approval UI, an email, a Slack message — and back. It's the join key of the whole pattern.
3. **Not re-running completed work.** Resuming must continue from the pause point, not replay from the start. Replaying is worse than a bug here: a re-run re-executes any side effect that already happened before the gate.

**Idempotency is not optional at a gate.** A human clicks Approve twice; the webhook is delivered twice; someone retries a stalled request. Each of those must produce one action, which is why [Stage 13's idempotency keys](stage-13-deployment.md) belong on exactly the side-effecting tools this stage gates. The two stages are describing one problem from two directions.

## The part nobody mentions: humans are slow, and gates queue

A pause isn't free, and this is where human-in-the-loop designs actually fail in production:

- **A pause can last days.** A gate opened Friday evening is answered Monday. Any in-memory state, open connection, or held lock is long gone — which is the practical argument for [durable execution](stage-13-deployment.md#agent-native-infrastructure--what-agents-need-that-ordinary-services-dont) once waits outlive a deploy cycle.
- **Every pause needs a timeout policy, decided in advance.** What happens to a gate nobody answers in 72 hours? Auto-reject, escalate, or expire? "It waits forever" is a decision too — usually the wrong one, and usually made by not deciding.
- **Approval fatigue is a real failure mode.** A human asked to approve 200 near-identical actions a day approves them all without reading — and your gate is now a very expensive no-op. **Gate quality beats gate quantity:** fewer, genuinely high-stakes gates get read; many low-stakes ones train people to rubber-stamp.
- **The reviewer needs enough context to decide.** "Approve refund: $45?" is unanswerable. "Approve $45 refund to customer #8812 — order delivered damaged, photo attached, customer's first refund" is answerable. If the gate doesn't carry the reasoning, the human is a rubber stamp by construction, however conscientious they are.

The design target that follows from all four: **gate rarely, and make each gate worth reading.** An agent that asks about everything and an agent that asks about nothing fail the same way — nobody is actually reviewing anything.

## Sources

| Type | Resource |
|------|----------|
| Docs | [LangGraph — Human-in-the-loop / interrupts](https://docs.langchain.com/oss/python/langgraph/human-in-the-loop) |
| Docs | [LangSmith — run review & audit trail UI](https://docs.langchain.com/langsmith/observability) |
| Pattern | Saga / checkpoint pattern for resumable long-running workflows (Temporal docs are a good non-LLM reference) |

## Done when

You can kill the process mid-run at an approval gate, restart it, and it resumes without re-running completed side effects.

---
[← Stage 08 — Multi-Agent Orchestration](stage-08-multi-agent.md) · [Back to roadmap](../README.md) · **Next:** [Stage 10 — Evaluation + Quality Assurance →](stage-10-evaluation-qa.md)
