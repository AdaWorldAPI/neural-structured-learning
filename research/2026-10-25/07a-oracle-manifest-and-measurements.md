# H7 / Reproducible comparison harness: oracle manifest v0

**Status:** draft preregistration; no tests run in this proposal. Companion to [PR #7](https://github.com/AdaWorldAPI/neural-structured-learning/pull/7).

## Experimental truth hierarchy

1. *Semantic oracle:* unmodified NSL behavior or independently defined mathematical result.
2. *Numerical oracle:* reference implementation / golden test vector generated without reusing our implementation code, e.g. TFLM C++ for #375.
3. *Performance baseline:* well-implemented alternative under the same input and hardware assumptions.
4. *Ablation:* intentionally disabled condition that **must fail** the corresponding check.
5. *Claim:* only the conjunction justified by evidence. Accuracy alone never demonstrates causal reasoning; CI green alone never demonstrates latency advantage.

## Manifest, one record per experiment configuration

~~~yaml
experiment_id: NSL-DR-001
pr_id: 1
hypothesis_revision: sha256:...
dataset_name: synthetic-directed-graph
dataset_sha256: ...
source_repos:
  nsl: <commit sha>
  lance_graph: <commit sha>
  ndarray: <commit sha>
  tensorflow: <version/commit>
references:
  onnxruntime: <version or not used>
  tflite_micro: <commit or not used>
codegen:
  rustc: <version>
  target_triple: <triple>
  compile_flags: <flags>
  simd_backend: <selected backend>
  rounding_policy: <policy>
environment:
  cpu_model: <model>
  cpu_flags: <explicit list>
  os_kernel: <version>
  memory_limit_bytes: <bytes>
  board_name: <MCU name or not used>
  flash_limit_bytes: <bytes or not used>
  ram_limit_bytes: <bytes or not used>
randomness:
  seeds: [11, 23, 37, 41, 53]
measurements:
  warmup_iterations: <n>
  samples_per_seed: <n>
  raw_csv_sha256: ...
oracle:
  script_sha256: ...
  output_sha256: ...
  tolerance_policy: exact_bits | ulp_bound | bounded_error | statistical
  confidence_interval_method: <method>
negative_controls:
  required: [<fixture identifiers>]
~~~

The example fields are **placeholders**, not actual recorded commits, CPU or test measurements. Any test runner must reject unfilled required fields instead of publishing a success badge.

## Dataset/task table

| Task | Input | Primary comparison | Secondary |
| --- | --- | --- | --- |
| B01 | NSL Cora graph tutorial | final accuracy and graph-loss distribution over pinned seeds | neighbor retrieval loss, runtime |
| B02 | NSL MNIST adversarial tutorial | loss/robustness under *same configured* attack | gradient, projection, perturbation norms |
| B03 | deterministic 12-edge directed graph | packed NL_* features vs focused references exactly | retained bytes, gather latency |
| B04 | tiny f32 ONNX MatMul+Add+Tanh | op-by-op output and max ULP | graph import and inference latency |
| B05 | TFLM int8 requant fixtures | exact bytes with pinned rounding regime | cycles and code size |
| B06 | 256-codebook sphere relation | certified LUT code, angular error | flash/RAM/energy |
| B07 | DR³ 0.01/1/50/100% occupancy | relational query results vs triple-set oracle | memory and access p95 |
| B08 | duplicate-source provenance fixture | correct refusal / shared-source lineage | trace size and replay latency |
| B09 | analytic diffusion/spring perturbation | deviation vs exact/controlled numerical reference | conservation and long-horizon error |

Cora/MNIST training accuracy depends on TensorFlow version, hardware, seed, dataset processing and nondeterministic kernels. Use distributional statistics and document non-determinism; do not require bit-parity for independent training runs.

## Timing / memory protocol

- All primary latency comparisons on the same physical device and power/thermal envelope. Predeclare whether allocation, import, serialization, data load and transfer overhead are included.
- Record *cold* and *warm* scenarios separately. No warmed focus cache on one arm and cold graph rebuild on another.
- Use repeated paired measurements with controlled random execution order, warmup, bootstrap intervals and raw samples. Report uncertainty and negative outcomes.
- Distinguish flash/ROM and RAM on MCU. Measure table residency and paging, not merely array bytes.
- If only simulation of a device is possible, label it "emulated," never claim hardware energy or hardware cycles.

## Mandatory negative controls

- Collapse duplicate directed edges by minimum, rather than maximum: graph oracle must fail.
- Reverse neighbor tie-break order: packed-graph oracle must fail.
- Bypass SourceId/index_seal validation: stale map must be refused by baseline check, and mutated version must be caught.
- Replace fused multiply-add by separate multiply+add on a known midpoint: bit parity gate must fail.
- Switch TFLM requantizer default double rounding to single rounding on a differentiating witness: parity gate must fail.
- Promote an adversarial correlation edge into a causal intervention: epistemic/causal test must refuse.
- Inject unsupported ONNX op and an unverified LUT rounding tie: compilation/load must refuse rather than fallback.
- Reuse same source evidence family twice: independent-evidence counter must stay one.

## Result artifacts (future, if/when experiments actually run)

- immutable manifest.yaml, commands.sh, golden vectors and input SHA list;
- tests.junit.xml or equivalent failure report;
- raw per-sample latencies and resident/flash byte counts;
- summary with confidence intervals, hardware fingerprint, known limitations;
- explicit *supported / bounded / refused / rejected* state per operator/experiment.

A research conclusion may be "baseline wins." That's a useful result, not a failure of recordkeeping.
