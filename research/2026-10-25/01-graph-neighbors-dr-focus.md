# NSL / DR focus: graph neighbors without materializing repeated features

Status: RESEARCH PROPOSAL, unimplemented. Scope: independent, falsifiable adapter experiment.

## Existing reference points
- NSL `tools/build_graph.py`: constructs weighted similarity edges from embeddings, including L2 normalization, cosine threshold, LSH buckets.
- NSL `tools/pack_nbrs.py`: writes augmented TFRecords with neighbor features and weights, limited by max_neighbors.
- NSL `keras/graph_regularization.py`: trains on neighbor logits; prediction/evaluation is delegated to base model.
- lance-graph #1762 (merged): experimental FocusManager lease keyed by full SourceId, MapId and Scope, gathering facet i32 values once. It is a workspace-excluded probe; no production graph-regularizer API.

## Hypothesis
Instead of copying neighbor features into each seed record, preserve graph edges as stable ordinals/relations with SoA facets, and gather only the active focus. Preserve the NSL loss semantics, including directed edges, weights, repeated neighbors, top-k, tie-break and missing-neighbor rules. DR64/DR256 are candidate mask/address structures, NOT assumed universal graph storage.

## Experiment
1. Extract tiny deterministic graph/TFRecord fixture from NSL's own graph and pack tests, and a synthetic high-degree graph.
2. Specify a neutral interchange: seed ID, neighbor ID, directed weighted edge, source revision and optional labeled/unlabeled status. Keep NSL's serialization untouched.
3. Implement an isolated Rust prototype that maps IDs to sealed ordinals, stores SoA features, and gathers active neighbor rows under a lease; no runtime adapter in this PR.
4. Compare generated neighbor batches and graph loss against unmodified NSL. Track zero-neighbor, duplicate edge, invalid ordinal, stale index seal and concurrency.
5. Measure peak bytes, cold-start gather latency, repeated-focus latency and feature duplication as graph density increases.

## Admission / falsifiers
- Exact neighbor identity, order and weights vs NSL oracle at each seed.
- Equal loss within a declared floating-point tolerance, with dtype/order stated; no blanket bit parity claim for TensorFlow.
- Refuse stale seals and out-of-range ordinals BEFORE any gather/prefetch.
- Compare cached and uncached paths; report cases where the focus lease loses.
- Disable test for seal check, and anti-vacuity fixture where same feature is used by >1 seed.

## Non-goals / boundary
No claim that #1762 is production ready, that a DR³ layout exists, or that graph regularization proves causality. Do not modify TensorFlow runtime or teach graph construction to infer biomedical mechanisms.

References: https://github.com/tensorflow/neural-structured-learning ; https://github.com/AdaWorldAPI/lance-graph/pull/1762
