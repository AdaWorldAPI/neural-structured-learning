# The steel thread: one falsifiable NSL → DR/SoA → inference → evidence demonstration

**Status:** design-specification v0. No integrated implementation or new runtime benchmark is claimed.
**Related:** [central scratchpad](00-brainstorming-scratchpad.md), [#1 graph focus](https://github.com/AdaWorldAPI/neural-structured-learning/pull/1), [#3 evidence](https://github.com/AdaWorldAPI/neural-structured-learning/pull/3), [#4 model DTO](https://github.com/AdaWorldAPI/neural-structured-learning/pull/4), [#7 benchmark](https://github.com/AdaWorldAPI/neural-structured-learning/pull/7).

## 1. The question worth betting a tiny experiment on

> Can we process the **same identified relations** as NSL training neighbors, as queryable inference-time structure, and as provenance-bearing reasoning evidence, while evaluating one frozen neural model through a **separately verifiable** native operator pipeline?

Three results would make this a useful experimental milestone:
1. The original TensorFlow NSL training neighbor semantics remain unchanged.
2. Native operator evaluation is numerically compatible with a pinned ONNX/TFLite (or TensorFlow) reference for a small, explicit subset.
3. An inference-time structural query and immutable evidence trace can be produced without changing the original neural prediction.

The **combined result**, not the breadth of imported technologies, is the scientific claim. Failure to achieve all three still yields useful independent findings.

## 2. A typed boundary, not a tensor-shaped mythology

~~~text
SOURCE RECORDS (observations, IDs, metadata)
  ├── NSL training graph edge: sample A -> neighbor C, weight 0.9
  ├── Computational dependency: MatMul -> Add -> Tanh
  ├── Logical premise: an explicitly typed ontology relation
  ├── Event lineage: output derived from model M, source S, revision V
  └── Mechanistic causal arrow: ONLY when a structural model justifies it
             │
             ▼
  typed identifiers + stable graph digests + declared semantics
             │
             ▼
  candidate shared substrate:
      DR64 / DR256 address/Boolean masks
      SoA typed values; ndarray SIMD scalar/vector operators
      RelAlg for relational queries
      optional DR³ paging, only if H6 evidence supports it
             │
        ┌────┴────────────┐
        ▼                 ▼
  numeric inference    structural/epistemic sidecar
        │                 │
        └──────┬──────────┘
               ▼
         typed result + provenance
~~~

**Non-equivalence constraints:**
- Data-neighbor graph edge ≠ executable neural graph edge.
- Executable neural edge ≠ causal/mechanistic arrow.
- Semantic ontology relation ≠ model-generated similarity weight.
- A high model score ≠ checked logical theorem.
- An authenticated event digest ≠ independent proof checker.
- Physical x/y/z coordinate axes ≠ arbitrary tensor rank dimensions.

Addressing can unify access without erasing these distinctions.

## 3. Frozen three-node toy experiment

Declare three source examples with two f32 features:
- A = [1.0, 0.0], B = [0.0, 1.0], C = [1.0, 1.0].
- Directed graph edges: A→C (0.9), A→B (0.5), B→C (0.4); C has no outgoing neighbors.
- max_nbrs=1, choose source-edge semantics **after** graph_utils duplicate-maximum canonicalization.
- Expected packed neighbor IDs: A→[C], B→[C], C→[].
- NSL's packing copies C's features into two separate examples; a candidate SoA design can store C once and reference it twice.
- Freeze an independent *graph-construction* seed and input order. Do not assert the LSH candidate builder is deterministic without controlling NumPy seed/set iteration and bucket order.

A deliberately tiny computational graph:
~~~text
f32[batch,2] --MatMul(identity f32[2,2])--> Add([0,0])
               --> Tanh --> output logits/embeddings f32[batch,2]
~~~
Expected real-valued mathematical oracle:
- t = tanh(1) ≈ 0.7615941559557649 (round to f32 under the chosen reference evaluator).
- h(A)=[t,0], h(B)=[0,t], h(C)=[t,t].
- A scalar **illustration**, not a claim about NSL's built-in reduction: define local edge score s(A,C)=0.9·||h(A)-h(C)||² = 0.9·t² ≈ 0.5220230925473764.
- Real NSL compatibility must use its configured DistanceConfig, transform, weights, axis and reduction, and compare to the pinned TensorFlow result. Do not hardcode this illustrative score as an NSL golden loss.

A focused execution:
1. Build directed graph from the exact source fixture, no silent undirecting.
2. Return the exact two selected neighbor batches using the original NSL tie/dedup rules; verify copied NL_* representation.
3. Decode the small model into a typed DTO; reject unsupported operators, shape/dtype/dynamic dimension, and quantization.
4. Evaluate outputs with independent ONNX Runtime reference and only those native kernels whose contracts are established (#1758 Tanh bounded alias; numeric MatMul/Add capability to be verified).
5. Reuse the **same source identity** for C across both neighbor queries; a second reference to C must not create a second independent observation.
6. Emit separate output event and provenance sidecar. With the sidecar disabled, the base-model output must remain unchanged.
7. Change SourceId(index_seal) or model graph digest; stale cache and replay must refuse.

### A minimal expected trace shape

~~~text
PredictionEvent { sample: A, model_digest: M1, output: [t,0],
                  numeric_contract: UlpBounded(2) }
NeighborSelection { seed: A, target: C, weight: 0.9,
                    graph_source: S1, source_revision: V1 }
NeighborSelection { seed: B, target: C, weight: 0.4,
                    graph_source: S1, source_revision: V1 }
EvidenceGroup { family: S1, referenced_by: [A,C; B,C], unique_sources: 1 }
ReplayAttempt { source_revision: V2, cached_V1: true, status: REFUSE }
~~~

The pseudo-records illustrate expected *properties*, not existing API types, certified proofs or actual run output. The source family is deliberately the same in the duplicate-evidence case.

## 4. Four independent checks, not one self-confirming harness

| Gate | Reference/oracle | Outcome required |
| --- | --- | --- |
| Structural | Original NSL graph_utils + pack_nbrs | same directed edges, max-weight duplicate rule, descending (weight, ID), features |
| Numerical | ONNX Runtime / TensorFlow CPU and exact mathematical special-value fixtures | per-node dtype+shape+value compatibility under declared exact or ULP bound |
| Referential | Independent immutable source/version/map table | stale mapping refused, reused C not doubled |
| Epistemic | Independent rule/proof checker where applicable | candidate remains candidate; no proof claim without admissible law + checker |

**Anti-vacuity mutants:** flip neighbor tiebreak, remove stale-seal check, replace FMA with a known double-rounding witness, double-count source C, promote a similarity edge to causal. Each should fail its relevant gate. A check that stays green under mutation is not evidence.

## 5. Work packages and ordering

- **W0: Independent fixtures and reproducible NSL oracle** from #7. Freeze scripts, input digests, build environment, seeds, datasets, and reference outputs.
- **W1: Neighbor crosswalk** from #1. Rust focus vs straight SoA hash map vs original packer. Correctness comes before throughput.
- **W2: Numeric DTO evaluator** from #4. One f32 graph first, then explicit int8/gemmlowp bridge after the missing i8×i8 zero-point accumulator is addressed.
- **W3: Optional structural sidecar** from #3. Keep plain inference fast and comparable, then measure evidentiary cost.
- **W4: Orthogonal excursions** from #2, #5 and #6: spherical perturbation and causal intervention, certified LUTs, DR³ tile layouts. They should not be prerequisites for W0-W3.

Every work package can report REJECTED when its measured candidate loses to a simpler equivalent baseline. Cross-repo implementation should be assigned to the layer that actually owns it: NSL fork for reference examples, Lance-graph for relation/adapter experiments, ndarray for verified primitives. The *research docs* live together here; speculative production APIs do not.

## 6. Questions that must remain open

- Does retaining structure provide **measurable extra utility** at inference, beyond retrieval after inference?
- Are graph-neighbor feature copies material at our scales, or is graph-loss computation the true bottleneck?
- Can we preserve useful learnable structure without trusting similarity edges as mechanistic causes?
- Does DR³ have a density regime where it beats both dense plane and sparse adjacency baselines?
- Does the pair-LUT help on a named MCU **after** flash paging and codebook quantization error are included?
- Does a typed operator DTO survive ONNX/TFLite graph versions, broadcasting, dtypes and quantization without reinventing an opaque tensor runtime?

If this thin slice succeeds, the next grant claim is modest but distinctive: *a tested relational execution and provenance bridge for structured biomedical AI*. It would not constitute AGI, clinical efficacy, or general tensor replacement.

## 7. Progress honesty

This page specifies an experiment. It contains no claim that the integrated run has been implemented, compiled, profiled or clinically evaluated.
