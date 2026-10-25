# H1 / Executable experiment contract: NSL neighbors to sealed DR focus

**Status:** proposed compatibility contract; no implementation or measured speedup. Companion to [PR #1](https://github.com/AdaWorldAPI/neural-structured-learning/pull/1).

## Source-level semantics to preserve

Read [graph_utils.py](../../neural_structured_learning/tools/graph_utils.py), [pack_nbrs.py](../../neural_structured_learning/tools/pack_nbrs.py), [build_graph.py](../../neural_structured_learning/tools/build_graph.py) and [graph_regularization.py](../../neural_structured_learning/keras/graph_regularization.py).

1. A graph is directed, keyed by source ID and target ID. The TSV parser uses edge weight 1.0 when omitted. Multiple source-target entries collapse to the **maximum** weight. Explicitly symmetrizing the graph is a separate option; do not assume edges are bidirectional.
2. Packing chooses outgoing neighbors that actually have a labeled or unlabeled Example, **sorts (weight, neighbor-ID) tuples in reverse order** and slices to max_nbrs; equal weight means lexicographically larger ID first. If the same ID is present in both tables, labeled seed features win.
3. A copied record contains NL_num_nbrs, NL_nbr_N_weight and NL_nbr_N_<feature-name> for every selected neighbor. Never treat absence as a zero-weight invented neighbor. Zero neighbors must retain an explicit zero count.
4. The graph builder's LSH buckets create candidate pairs, not an exact all-pairs nearest-neighbor guarantee when LSH is enabled. Its generated edges use L2-normalized embedding dot products. Comparing against an all-pairs oracle is a *different* experiment.
5. graph_regularization.py applies the graph loss to **neighbor logits during training**; predict and evaluate delegate to the base model. Preserve loss function, reduction axis, weights and graph multiplier. Setting no neighbors or multiplier <=0 must result in zero added loss.
6. If the TSV graph was written by graph_utils.py, weights are formatted using %f; compare after the same serialization/parse pipeline rather than assuming the source embedding float survives bit-identically.

## Proposed read-only adapter (not actual Lance APIs)

Input:
- immutable SourceRef { dataset_id, branch_id, version, index_seal }, scope and MapId;
- interned seed_ID / neighbor_ID dictionaries; immutable directed edges (src_ordinal, dst_ordinal, f32 weight);
- typed SoA facet columns and declared per-feature lengths, null/absence policy;
- requested neighbor cap (None or nonnegative integer), query fingerprint.

Output:
- NeighborBatch { seed_id, selected: ordered [ {neighbor_id, weight, source_id} ], view/references to typed feature columns };
- a compatibility serialization procedure for producing the original NL_* TF Example oracle;
- no implicit ID coercion, no NaN weight silently sorted, no cross-branch map reuse.

**Scope note:** #1762 tested gathered i32 facet values under SourceId+MapId+Scope lease. A graph-loss wrapper needs float/logit and feature handling that does **not** exist merely because #1762 merged. Treat that work as a reusable experiment pattern, not a ready API.

## Fixture table (write and freeze before any optimization)

| Fixture | Graph input | Observable |
| --- | --- | --- |
| G01 | A->B .2, A->C .7 | C then B |
| G02 | A->B .4 and A->B .9 | exactly one B at .9 |
| G03 | A->B .9, A->C .9 | C before B (reverse ID tiebreak) |
| G04 | A->missing .99, A->B .1 | missing filtered, B selected |
| G05 | B exists in both seed and unlabeled dictionaries | seed features take precedence |
| G06 | A has no outgoing neighbors | NL_num_nbrs=0 and no NL_nbr_0_* |
| G07 | A->B only, no symmetrization | B does not gain A as neighbor |
| G08 | same seed/feature referenced by >=3 graph seeds | oracle parity, measure avoided bytes |
| G09 | same ordinal map with changed index_seal | refuse before feature read |
| G10 | same key concurrent load (>=8 readers) | exactly one gather if lease is shared |
| G11 | cap 0, cap 1 and unlimited | correct truncation, no invented rows |
| G12 | negative, NaN or infinite edge weights | explicit documented handling; do not derive a silently different total order from Python |

Rows G01-G11 are *proposed golden test inputs*, not asserted test outcomes. For float tie tests, serialize/parse identically on both sides. The source implementation's behavior for pathological NaN/Inf ordering needs an explicit decision, not a presumed equivalence.

## Comparison protocol

- **Functional primary endpoint:** exact chosen (seed_ID, neighbor_ID, order, weight f32 bits after TSV parse) and copied NL_* feature values; graph-loss tolerance set per reduction and dtype, recording each mismatch.
- **Cost endpoints:** packed TFRecord bytes; edge/ordinal/SoA/map bytes; peak resident memory; cold and warm gather latency p50/p95/p99; cache invalidation work; total sample preparation wall-time.
- **Controls:** unmodified NSL, an uncomplicated Rust HashMap+SoA gather, and #1762-inspired sealed focus. Prefer simple path if focus adds cost.
- **Workloads:** uniform and power-law degree distributions, isolated vertices, many shared neighbor features, maximal cap, high churn vs read-mostly.
- **Disable/negative gates:** remove dedup-max; reverse tie rule; suppress seal check; bypass lease; all must trip their corresponding fixture.

## Review boundary

A passing adapter proves *behavioral equivalence of neighbor retrieval*, not clinical validity, speedup, or causality. Only after G01-G12 and memory/timing data should a implementation PR be opened in Lance-graph or a separate Rust experiment crate.
