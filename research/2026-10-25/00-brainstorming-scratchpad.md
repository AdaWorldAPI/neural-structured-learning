# Research scratchpad: Neural Structured Learning × Lance-graph × ndarray

**Date:** 2026-10-25  
**Status:** BRAINSTORMING / PROPOSED; not an implementation or a benchmark result.  
**Working theory:** structured training signals (graph and adversarial perturbations) can potentially be retained as addressable, verifiable computational relations through inference, rather than discarded once a neural network is trained. An ONNX/TFLite interchange should be decoded at the adapter boundary, not force a persistent tensor object model into Lance-graph. The outcome must be measured, not presumed.

## 1. Starting evidence, not mythology

| Reference | Verified in reviewed PR/source | What it does **not** establish |
| --- | --- | --- |
| [TensorFlow NSL](https://www.tensorflow.org/neural_structured_learning) | Explicit graph and implicit adversarial structure improve training in supported workflows; standard inference does not require structure | NSL is not a general-purpose relational reasoning engine |
| [NSL build_graph.py](https://github.com/tensorflow/neural-structured-learning/blob/master/neural_structured_learning/tools/build_graph.py) | L2-normalized embeddings, cosine similarity, LSH-based candidate graph building | Does not imply an acos/cosine roundtrip is happening in every TFLite kernel |
| [NSL pack_nbrs.py](https://github.com/tensorflow/neural-structured-learning/blob/master/neural_structured_learning/tools/pack_nbrs.py) | Neighbor feature materialization into TFRecords with weights and deterministic selection rules | Does not imply such materialization is always the bottleneck |
| [NSL graph_regularization.py](https://github.com/tensorflow/neural-structured-learning/blob/master/neural_structured_learning/keras/graph_regularization.py) | Additional training objective for neighbor logits; direct base-model inference | Does not guarantee full train/export parity under ONNX/TFLite |
| [NSL adversarial_regularization.py](https://github.com/tensorflow/neural-structured-learning/blob/master/neural_structured_learning/keras/adversarial_regularization.py) | Perturbations by loss gradients, with feature configuration | No causal intervention semantics |
| [lance-graph #1758](https://github.com/AdaWorldAPI/lance-graph/pull/1758) | Merged native ULP-bounded ONNX aliases for Tanh/Exp | Not a complete model importer/runner |
| [lance-graph #1762](https://github.com/AdaWorldAPI/lance-graph/pull/1762) | Merged lab: sealed focus lease and gathered facet; no timings | Not DR³ nor production readiness |
| [ndarray #370](https://github.com/AdaWorldAPI/ndarray/pull/370) | Merged generic compile-time Codec8 scheme | Not a complete int8 inference engine |
| [ndarray #371](https://github.com/AdaWorldAPI/ndarray/pull/371) | Closed **unmerged** WASM rounding proposal | Must not treat draft optimizations as shipped |
| [ndarray #372](https://github.com/AdaWorldAPI/ndarray/pull/372), [#373](https://github.com/AdaWorldAPI/ndarray/pull/373) | Merged saturating narrow ladder, with #373 correcting the landing | Do not assume every revision of #372 landed |
| [ndarray #374](https://github.com/AdaWorldAPI/ndarray/pull/374) | Merged one-rounding FMA on backend paths | Do not assume every transcendental is correctly rounded |
| [ndarray #375](https://github.com/AdaWorldAPI/ndarray/pull/375) | Merged gemmlowp/TFLM fixed-point requantization, with independent golden data | No int8×int8 GEMM with zero points and no LUT activations in that PR |
| [tensorflow/rust](https://github.com/tensorflow/rust) | Archived March 17 2026; Rust TensorFlow C API bindings | Not a live dependency strategy |

## 2. Hypothesis map / independent draft PRs

| ID | Draft PR | Question | First falsifier / evidence |
| --- | --- | --- | --- |
| H1 | [#1: Graph focus](https://github.com/AdaWorldAPI/neural-structured-learning/pull/1) | Can sealed DR/SoA focus replace copied neighbor features without changing NSL neighbor semantics? | Exact neighbor selection/weights against unmodified NSL; stale seal refuses |
| H2 | [#2: Perturbation](https://github.com/AdaWorldAPI/neural-structured-learning/pull/2) | Can adversarial, geometric and intervention perturbations share a descriptor while retaining distinct meanings? | Coordinate seams, gradient comparison and explicit causal negative control |
| H3 | [#3: Persistent evidence](https://github.com/AdaWorldAPI/neural-structured-learning/pull/3) | Can graph evidence remain queryable at inference without altering neural output? | Identical disabled output, source de-duplication, stale replay refusal |
| H4 | [#4: Model DTO](https://github.com/AdaWorldAPI/neural-structured-learning/pull/4) | Can a pinned ONNX/TFLite graph lower into DR+SoA+RelAlg+SIMD on a restricted operator set? | Independent runtime parity with explicit unsupported-op refusal |
| H5 | [#5: Pairwise LUT](https://github.com/AdaWorldAPI/neural-structured-learning/pull/5) | Does directly tabulating a finite pairwise geometric operation outperform trigonometric or cosine baseline? | Full vs tiled LUT vs real baseline across error, latency and memory |
| H6 | [#6: DR³ hypercube](https://github.com/AdaWorldAPI/neural-structured-learning/pull/6) | Is a 3D address layout better than CSR/COO and SoA for sparse model/ontology/temporal relations? | Lossless addressing, worst-case memory, cache costs, update correctness |
| H7 | [#7: Baselines](https://github.com/AdaWorldAPI/neural-structured-learning/pull/7) | Are the comparisons reproducible and genuinely equivalent? | Frozen NSL Cora/MNIST and independent numerical oracles |

All seven are DOCUMENTATION-ONLY research PRs, NOT merged code, deployed adapters or already validated performance gains. Implement in appropriate repository after design review; do not scatter speculative functionality across upstream TensorFlow repositories.

## 3. The core synthesis, with deliberately hard boundaries

```text
NSL explicit graph ---------------> weighted adjacency / evidence IDs
NSL adversarial examples ---------> gradient-perturbation descriptor
Geometry / sphere / Helix --------> charted coordinate perturbation (separate semantics)
ONNX / TFLite --------------------> pinned graph DTO: nodes, edges, dtypes,
                                     shapes, constants, quantization contract
                    |                         |
                    +-------------+-----------+
                                  v
          candidate substrate: DR64/DR256 mask/focus + SoA values
                         + RelAlg + ndarray SIMD
                       /           |            \
              model evaluation   graph querying  candidate logic
                   |                 |               |
                   +---------- provenance ------------+
                                   |
                              temporal replay
```

**Important negative assertions:** a neural model graph is not itself a knowledge graph; graph similarity is not causality; a probability is not a proof; DR³ is not interchangeable with arbitrary tensor ranks; a LUT needs a mathematically specified operation and domain; a proof capsule is not independently certified merely because it has a digest.

## 4. High-leverage experiments, ranked by information gain (not promised feasibility)

**E0. Replicate truth.** Freeze NSL's Cora graph and MNIST adversarial tutorials (H7). Preserve Python/TF versions, seeds, tolerance and raw results. No large new code dependency yet.

**E1. Graph adjacency vs gathered focus (H1).** Join `pack_nbrs` oracle to the #1762 lab's lease semantics. Track directed edges, top-K, weights, duplicate and missing neighbors, sealed revisions. Compare equivalent graph loss; run cold/warm focus memory and timing.

**E2. Small ONNX and TFLite graph DTO (H4).** Reconstruct one micro-model (input → matmul/add → activation). Confirm exact or ULP-bounded output against pinned reference, and TFLM default double-rounding requant. Refuse unsupported op/dtype/version. Do NOT interpret safetensors/GGUF as executable models.

**E3. Mathematical pair-LUT probe (H5).** On a defined, discrete sphere-direction operation, high-precision table generation and certification, 256² table vs focused/factorized table and real analytic baseline. 256² u8 entries are 64 KiB, frequently too large for small MCU RAM; separately measure ROM, RAM and cycles on named board. Compare multiple composition steps for accumulated drift.

**E4. Disentangle perturbation (H2).** Contrast gradient-based adversarial NSL examples with coordinate perturbations and explicitly modeled interventions, including two deliberate confusion/negative tests.

**E5. Evidence channel (H3).** Same neural output with evidence disabled; present proof/refusal states distinct from uncertain beliefs. Duplicate-source replay must not double count.

**E6. Hypercube benchmark (H6).** 64³ = 262,144 logical cells, 32 KiB dense at one bit/cell; 256³ = 16,777,216 cells, 2 MiB dense. Compare to 2D planes (256² = 8 KiB). Values/indices/cache overhead separate. Try realistic skewed data where hypercube could lose badly.

## 5. Decision gates and ownership

- Do **not** merge implementation before an independent parity oracle and explicit refusal tests.
- Original upstream NSL is Python/TF. Proposed Rust experiment can be external, cross-repo, and benchmark-based. Never introduce a TensorFlow C API runtime dependency merely because tensorflow/rust exists; that repository is archived.
- Preserve numeric contracts: #1758 bounded transcendental aliases, #374 exact FMA, #375 TFLM quantization rounding mode, #370 Codec8 code spaces. Never replace these by undocumented approximations.
- No silent clinical validation claims. Synthetic data and public graph benchmarks first, then external clinical supervision and ethical review if pursued for MedCare-rs.
- Open draft hypotheses independently; once falsified, record rejection rather than rewriting history. Cross-repo implementations belong in lance-graph/ndarray only when validated.
- A Horizon healthcare AGI / community-infrastructure application can describe these as a **research agenda** with falsifiers, not a completed AGI system.

## 6. What could make this original?

NSL applies structure as a *training signal*. The proposed novelty is structure as a *persisting addressable computational object* across model execution, relational reasoning, perturbation experiments and provenance. This has not been demonstrated by the source repositories reviewed.

A successful small demonstration would show: same trained neural model result, an independently useful relational neighborhood query, and an evidence-preserving temporal trace, under a documented overhead budget. If the overhead dwarfs the benefit, preserve NSL's simpler training-only approach.

## 7. Evidence log

2026-10-25: audited documentation, NSL graph/regularizer/packing source, and specified lance-graph/ndarray PRs. Opened seven draft documentation proposals; **no code compiled, no experiment executed, and no performance measured as part of this scratchpad**.
