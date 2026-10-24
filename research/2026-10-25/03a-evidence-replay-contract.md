# H3 / Persistence of structure: an evidence-aware optional inference companion

**Status:** proposed contract, unimplemented. Companion to [PR #3](https://github.com/AdaWorldAPI/neural-structured-learning/pull/3).

## Distinguish three identity systems

1. **Referential identity:** source dataset/branch/version/index seal, entity/edge ID, scope, explicit mapping identity. This says *which thing* is being discussed.
2. **Temporal/causal-event identity:** operation ID, logical time/order, parents, rule/compiler revision and event digest. This says *which computational event depended on which event*, not that causal inference is scientifically justified.
3. **Epistemic identity:** observation, learned score, hypothesis, refuted statement, or independently checked theorem with certificate. This says *what support is claimed and what support is not present*.

These identifiers may coexist on a row but are not interchangeable. The existence of a temporal.rs module or a digest is not evidence that all three systems have been implemented.

## Proposed immutable event envelope

~~~text
EvidenceEvent {
  event_id: Hash(canonical_bytes),
  parent_event_ids: sorted Vec<EventId>,
  ref_id: SourceRef(dataset, branch, version, index_seal),
  map_id, scope, entity_ids, edge_ids,
  evidence_family_id: Option<SourceFamilyId>,
  kind: Observed | ModelScore | Hypothesis | Intervention |
        RuleApplication | Contradiction | Refusal | CheckedProof,
  model_digest?: Digest, rule_digest?: Digest,
  payload_digest: Digest,
  support_ids: sorted Vec<EventId>,
  epistemic_state: Candidate | Refuted | Certified | Unknown,
  proof_reference?: ProofRef,
  evaluation_contract: Exact | UlpBounded(N) | QuantizedByteParity |
                       StatisticalOnly
}
~~~

This is a **schema sketch**, not a suggested new public Lance-graph API. The actual proof checker, ABI and trust boundary must be inspected before any implementation.

## Invariants to prove or test

**R1 immutable source:** Change to source version, index seal, map ID or scope invalidates reuse of a cached relational observation. Never rebind a stale ordinal.

**R2 deduplication:** Two independent events citing the same observation family cannot count as two independent sources. A repeated calculation is also not new evidence.

**R3 temporal ordering:** Replay is stable for the same immutable input graph, rule revision and deterministic scheduler. Distinct wall-clock ordering is not necessarily logical causality.

**R4 epistemic refusal:** A prediction is a model output even with 99.9% probability. It is not a checked proof. A purported certificate must be validated by an independent checker and satisfy its admitted law and scope.

**R5 unchanged base path:** If the optional evidence sidecar is disabled, running NSL's base model should produce the ordinary inference output. Benchmark incremental cost of sidecar enabled, not just total latency.

**R6 secure provenance:** A source/digest can leak sensitive information through identifiers or trace payloads; clinical evidence requires privacy review. Use synthetic fixtures here.

## Canary examples

- Event A: one labeled source S supports an edge E. Event B: a second derived embedding links to E but ultimately cites the *same* S. A and B must not become two independent confirmations.
- Event C: another genuinely distinct source S2 supports E. The lineage should contain two source families, without claiming a proof of E.
- Event D: run under a new dataset version and reapply a stale MapId/index seal. Expect refusal, not successful cached lookup.
- Event E: logic rule R with checked proof certificate is replayed under an altered rule digest R2. The old proof is invalid until independently rechecked.
- Event F: predicted class Dog has low-confidence conflicting ontology evidence. Preserve both, never overwrite one with a categorical asserted truth.
- Event G: two branches produce incomparable assumptions. Merge only where an explicit rule permits; a majority vote is not a logical merger.

## Thin vertical slice

On a 3-node synthetic NSL graph, run one frozen model and:
1. store output only (baseline),
2. store output and sidecar edge provenance,
3. query one neighbor relation without rerunning the model,
4. replay with unchanged seals, then with a stale seal,
5. inject a duplicate source; check whether support lineage remains deduplicated,
6. generate human-readable explanation showing *observed graph evidence* versus *model score*.

Measure trace byte size / event, replay p50/p95, query overhead, false reuse rate (must be zero), and output parity with evidence disabled.

## Exclusions

This is not automatic XAI: a neighbor relation explains a recorded structural input, not the internal decision mechanism of a neural network. Nor is a proof ref a proof without a checker, and no amount of epistemic bookkeeping establishes physical causality.
