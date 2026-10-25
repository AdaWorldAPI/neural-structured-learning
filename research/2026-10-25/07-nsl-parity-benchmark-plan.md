# NSL cross-substrate benchmark: preregistered, independent comparison

Status: RESEARCH PROPOSAL, unimplemented.

## Why
A shared substrate claim must earn evidence for accuracy, speed, memory and interpretability separately. Avoid cherry-picking comparisons, implicit reliance on an unfrozen reference, or CI passes mistaken for clinical validation.

## Reference tasks
1. NSL natural graph tutorial (Cora): graph-regularized classifier with explicit neighbor features.
2. NSL adversarial tutorial (MNIST): adversarial perturbation during training.
3. Synthetic perturbation/sphere toy: geometry and exact analytic oracle.
4. Tiny exported neural model: frozen ONNX/TFLite outputs, with per-op dtype and quantization constraints.
5. Duplicate-source provenance toy: expected refusal under repeated evidence.
Use public/synthetic data only; do not import patients or protected clinical examples.

## Reproducibility contract
- Freeze repository revisions, Python/TensorFlow/NSL versions, test vectors and seeds; record hash manifests and training hyperparameters.
- Distinguish training accuracy and loss from inference byte/ULP parity. Floating training is not presumed deterministic across GPU/compiler variants.
- Pin CPU ISA (AVX2/AVX-512/NEON/WASM), Rust toolchain, memory limits, target board and thermal conditions; publish raw sample distributions.
- Pre-register baselines: unmodified NSL/TF, ONNX Runtime, TFLite Micro when applicable, CSR/COO graph path, dense or sparse DR.
- Do not include non-equivalent pipeline work in speed denominators without labeling it.
- Run failing negative controls: stale revision, invalid ordinal, duplicate causal evidence, unsupported ONNX op, adversarial vs causal confusion, incorrect rounding.
- Require an independent source/oracle for numerical parity (e.g. actual TFLite C++), not a copy of the implementation.

## Metrics
Task accuracy/F1 or loss with uncertainty; output bit parity or max ULP; graph-loss deviation; expected calibration error; angular drift; wall-clock latency percentiles; peak allocated RAM; Flash/ROM footprint; energy if available; trace bytes; cold/warm focus and refusal rates. Include confidence intervals and multiple seeds.

## Admission
Publish passing AND failing trials, environment manifest and exact commands. A failed candidate still counts as useful evidence; do not promote to 'supported' until gates are met. No clinical/AGI claim inferred from a synthetic benchmark.

References: https://github.com/tensorflow/neural-structured-learning ; https://github.com/AdaWorldAPI/ndarray/pull/374 ; https://github.com/AdaWorldAPI/ndarray/pull/375
