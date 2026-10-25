# H4 / Typed model graph DTO: operator semantics before optimization

**Status:** interface proposal and phased oracle plan, not implemented. Companion to [PR #4](https://github.com/AdaWorldAPI/neural-structured-learning/pull/4).

## What is being imported?

A computational graph is **not** automatically an ontology graph. The adapter decodes the executable format into explicit nodes, typed ports, producer/consumer edges, graph input/output identities, constants and operator/version contracts. Model bytes are not executable merely because weight data is present.

A prospective minimal intermediate form, written as conceptual types:

~~~text
ModelGraph {
  source_format: ONNX | TFLite,
  format_schema_version: String,
  opset_or_builtin_version: String,
  graph_digest: Hash,
  inputs: Vec<ValueId>, outputs: Vec<ValueId>,
  values: Map<ValueId, ValueSpec>, nodes: Vec<ComputeNode>
}
ValueSpec {
  dtype: F32 | I8 | U8 | I32 | ...,
  logical_shape: Vec<Dim>,  // Known(u64) or Symbolic(name)
  initializer_ref?: ImmutableBytesRef,
  quant?: QuantSpec
}
ComputeNode {
  id, domain, op_name, version, inputs: Vec<MaybeValueId>,
  outputs: Vec<ValueId>, attributes, source_order
}
QuantSpec {
  storage_dtype, scale: PerTensor(f32) | PerAxis(Vec<f32>, axis),
  zero_point: PerTensor(i32) | PerAxis(Vec<i32>, axis),
  rounding_mode, saturation_policy, accumulator_dtype
}
LoweredOp { operator_contract_id, operands, mask, destination, op_fidelity }
~~~

**Important:** These are *proposed contracts*, not existing Rust structs. Shapes and dtype are still necessary descriptors even if no reusable 'tensor object' survives the adapter. Arbitrary tensor rank needs coherent flattening or view semantics with a reversible index map. DR³ alone is not a substitute for this.

## Separate numerical guarantees

| Class | Examples | Contract |
| --- | --- | --- |
| Exact integer / Boolean | mask comparison, pure index mapping, bitwise ops | exact bits, explicit overflow mode |
| IEEE operation | FMA and precisely specified additions/reductions | reference rounding order, signed zero/NaN handling |
| Bounded transcendental | lance-graph #1758 Tanh and Exp aliases | measured ULP bound, exact special-value policy |
| Quantized arithmetic | ndarray #370 Codec8, #375 TFLM requantization | byte parity for pinned TFLM default *double rounding* oracle |
| Approximation refused | unsupported Log, Sigmoid or op-version/fusion | explicit REFUSE until bound/reference exists |
| Training-only | NSL graph regularization/adversarial optimizer | do not silently lower training logic into inference graph |

Never fuse two floating ops without demonstrating that re-association preserves the declared numerical contract. In particular, #374 FMA has **one rounding**, not the two roundings from separate Mul+Add.

## First tiny graph, fully inspectable

~~~text
input f32[1,4] --> MatMul(f32[4,3]) --> Add(bias f32[3])
               --> Tanh bounded --> output f32[1,3]
~~~

- Freeze input values with 0, -0, subnormal, one NaN, ±inf in **separate targeted fixtures** (avoid contaminating ordinary classifier test with all special values).
- Reference execution: pinned ONNX Runtime CPU, node outputs captured independently. Provenance includes model file SHA-256 and opset.
- Lower MatMul only for supported dimensions. Reject rank, shape, data layout or broadcasting cases not explicitly covered. Validate bias Add right-aligned broadcasting.
- Tanh must stay in a bounded/operator namespace and preserve special values. Record per-output maximum ULP vs independent reference, not only classifier label.
- Graph sorting detects unknown producers, cycles, missing initializer, duplicate output binding, unhandled optional inputs.
- Initially **do not claim general constant folding/fusion/autodiff/control flow**.

## Separate TFLite Micro int8 gate

~~~text
int8 activation + int8 weights + zero points
       -> int32 accumulator
       -> per-tensor/per-axis scale & offset
       -> gemmlowp/TFLM fixed-point requantization
       -> int8 output
~~~

ndarray #375 **does not provide an int8×int8 GEMM with zero-point handling**. It provides the requantization end of the chain, not the whole operator. Until a verified accumulate kernel exists, the experiment may run the reference TFLM accumulator and native #375 epilogue, or refuse complete native FullyConnected. Both outcomes must be clearly labeled.

Two rounding regimes in TFLM must be distinct: default double rounding and TFLITE_SINGLE_ROUNDING. Never compare values across modes as if they were the same contract.

## Admission matrix and deterministic test vectors

| ID | Fixture | Expectation |
| --- | --- | --- |
| O01 | valid 1x4/4x3 + 3 bias + Tanh | ordered graph and ULP-checked output |
| O02 | rank mismatch 1x5 vs 4x3 | explicit shape refusal |
| O03 | missing constant, unknown op, unknown opset | explicit refusal, no fallback-by-name |
| O04 | repeated initializer, shared weight | address aliasing does not duplicate physical bytes in manifest |
| O05 | +0 / -0, NaN, ±inf, subnormal | signed-zero and bounded-value oracle gate |
| O06 | quant zero-point at signed range limits | exact reference accumulator/epilogue separation |
| O07 | per-channel scales and axes | no confusion of data/channel dimension |
| O08 | default vs single-rounding mismatch witness | mode switch exposed and oracle-distinguishable |
| O09 | longer rank-4 Conv input (unsupported initially) | must refuse rather than wrongly flatten |
| O10 | unknown control-flow node | must refuse rather than erase control flow |

## Deliverables and stop rule

D0 manifest parser with stable graph digest and a table of all unsupported semantics.
D1 independent output vectors with scripts, pinned runtime versions and raw mismatches.
D2 tiny CPU execution path, no instrumentation hidden in benchmark critical loop.
D3 optional WASM / TFLM target-specific exploration after D2.

Performance measured only after passing O01-O10. The proof target is a *small certified compatible operator subset*, not "native TFLite in Lance-graph" in general.
