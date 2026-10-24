# ONNX / TFLite graphs as DTO boundaries, not a permanent tensor runtime

Status: RESEARCH PROPOSAL, unimplemented.

## Anchors
- lance-graph #1758 (merged): ONNX facade uses native ndarray Tanh/Exp kernels under an explicit bounded namespace (not graph lowering). No full model execution parity was claimed.
- ndarray #370 (merged): compile-time Codec8 types; #372/#373 saturating narrows; #374 single-rounding FMA; #375 TFLite/gemmlowp requantization.
- tensorflow/rust was archived 2026-03-17 and binds TensorFlow via its C API. Reference only, no new dependency.
- NSL's Keras wrappers are training-time Python/TensorFlow interfaces, not ONNX/TFLite exports by themselves.

## Hypothesis
A restricted, pinned ONNX or TFLite graph can be decoded to a neutral node/edge/shape/dtype/constant/operator/quantization DTO, then lowered to DR addressing + SoA facets + relational execution plus SIMD, without building an independent general tensor object model. The adapter must expose unsupported cases; formats are not mutually lossless.

## Phased experiment
S0: Choose one exported small classifier and exact ONNX opset / TFLite schema; archive hashes and reference runtimes.
S1: Decode input/output identities, topological order, constants, broadcast dimensions and quantization params; compare graph manifests.
S2: Lower only supported Add, MatMul/Gemm, Relu and representative reduction, leaving unsupported ops as explicit REFUSE.
S3: Execute single subgraph with side-by-side ONNX Runtime / TFLite or native TF references; define per-op exact vs ULP-bounded vs quantized byte parity.
S4: Add an NSL-trained toy model to test *training/export/inference* workflow; do not assume adversarial optimizer itself exports.
S5: Record SIMD CPU backends, WASM and bare-metal constraints separately; no dynamic function-pointer dispatch introduced solely for codec selection.

## Falsifiers / gates
- Dtype/shape/broadcast, initializer aliasing, operator versions, absent/optional inputs and quantization zero-points checked.
- FMA single-rounding and TFLM default double-rounding requant pinned; separate test for TFLITE_SINGLE_ROUNDING mode.
- Runtime source of truth remains independent and reports every mismatch, not only aggregate accuracy.
- Unknown operator/version MUST refuse; graph conversion cannot silently approximate or alter signed zero/NaN policy.

## Non-goals
Neither GGUF nor safetensors alone specifies an executable ONNX/TFLite graph. No claim that a neural graph automatically becomes Datalog logic or that DR³ is already validated.

References: https://github.com/AdaWorldAPI/lance-graph/pull/1758 ; https://github.com/AdaWorldAPI/ndarray/pull/375 ; https://github.com/tensorflow/rust
