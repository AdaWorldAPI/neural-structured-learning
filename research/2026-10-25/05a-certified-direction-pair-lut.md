# H5 / Certified pairwise lookup: which operation is actually tabulated?

**Status:** research protocol and candidate domains, unimplemented. Companion to [PR #5](https://github.com/AdaWorldAPI/neural-structured-learning/pull/5).

## Two fundamentally different tables

Let each code a ∈ {0,…,255} identify a fixed unit direction u_a ∈ R³ and b identify u_b. These direction vectors and the codebook hash are immutable.

**A. Cosine similarity table**
- C[a,b] = Q_c(u_a · u_b), where the real dot product is in [-1,1].
- This can be used as a *candidate* replacement for cosine evaluation when inputs are **already discretized directions** and the cosine threshold is reinterpreted under a documented quantization budget.
- It is not a drop-in equivalent for NSL's continuous L2-normalized embedding similarity, especially near a threshold.

**B. Angular separation table**
- A[a,b] = Q_theta(arccos(clamp(u_a·u_b,-1,1))), angle in [0,pi].
- This answers a different question: angular separation, not cosine weight. It may be useful for Helix/spherical operations and can avoid runtime arccos for this finite codebook.
- A and C are related mathematically, but quantization and inverse mapping are not generally lossless. Never confuse an angle with a directed rotation (an angle lacks rotational handedness/axis information).

**C. Composition is another table**
- A spherical rotation composition is not captured by a table of pairwise angular separations. To compose directed rotations, a codebook of orientations and a projection rule must be specified; the operation may be nonassociative after quantization and must be separately tested.

This avoids asserting that TensorFlow NSL's graph builder actually executes acos(cos(x)). It computes normalized embedding dot products; an acos roundtrip is not established by the reviewed code.

## Proposed code and metadata contract

~~~text
PairLutHeader {
  kind: CosineSimilarity | AngularSeparation | OrientationCompose,
  codebook_hash, generator_hash, math_reference,
  row_count: 256, column_count: 256,
  entry_type: I8 | U8 | U16, value_scale, value_zero_point,
  rounding_policy, NaN_policy, invalid_code_policy,
  max_certified_error, certified_version
}
lookup(a: u8, b: u8) -> code   // constant-time indexing; no runtime dispatch
~~~

This is a research DTO, not shipped ndarray API. TFLite quantized activation rules and geometric angle-code rules need not share the same code space. Use #370's marker-type architecture only where policy equivalence is proven. #375's requantization validates a **different stage** (accumulator-to-i8), not this LUT.

## Certification generator (offline, never runtime MPFR)

1. Pin exact codebook vectors: either rational coordinates, a high-precision construction with interval certificates, or explicitly rounded f32 triples whose intended mathematical interpretation is documented.
2. Compute rigorous intervals for dot product and, for angular table, acos. Distinguish errors introduced by input codebook quantization from error in the table's stored code.
3. For each [a,b], certify exactly one output integer after specified ties-even / ties-away / floor rounding. If the certified interval straddles an output tie, increase precision; if still unresolved, **fail generation**.
4. Generate immutable const array and a certificate manifest (generator commit, inputs, digest, per-entry error maximum, worst-case witnesses); independently re-evaluate a sampled/full subset using a different precision engine.
5. Query time must be an array lookup with explicit bounds/invalid policy. No trait objects, per-element function pointers, runtime lazy initialization, or hidden libm calls.

## Symmetry / invariants

Cosine and angular separation tables should be symmetric (A[a,b]=A[b,a]), and diagonals must reflect the codebook self-relation. Whether entries around antipodal directions satisfy exact identities depends on whether the codebook includes exact antipodal partners. Signed zero, invalid codes, phi seam, poles, and periodicity require targeted tests. Composition tables need separate identity/inverse/associativity drift tests.

## Memory arithmetic and deployment questions

| Representation | Raw table bytes | Extra constraints |
| --- | ---: | --- |
| 256×256 u8 | 65,536 (64 KiB) | excludes header and alignment |
| 256×256 u16 | 131,072 (128 KiB) | alignment and metadata add cost |
| 256×256 f32 | 262,144 (256 KiB) | full precision values, not compact codes |
| symmetric triangular u8 | 32,896 (32.125 KiB) | index transform and latency; only valid for symmetric function |
| 16×16 active u8 tile | 256 | fetching/selecting tiles has nonzero cost; full backing still needed somewhere |

These are storage lower bounds, **not** a statement that an Arduino can hold the table in RAM. For an Arduino Uno-class 2 KiB SRAM / 32 KiB flash target, the full u8 LUT does not fit internally even in flash; use a named larger board or an external storage strategy and measure costs. Report ROM vs RAM explicitly. Never assume that a fast desktop LUT outperforms arithmetic on cache-starved MCU.

## Head-to-head benchmark protocol

- Use exactly the same (a,b) coded inputs for direct dot product + angle, full LUT, packed triangle and tiled page lookup.
- Include threshold-near pairs, codebook neighbors, antipodes, seams/poles, random pairs, and chains of 1/10/100 compositions where meaningful.
- Baseline = actual chosen operation implemented numerically, not a fabricated acos(cos(x)) path.
- Report average / p95 / max angular error in radians, boundary misclassifications for cosine threshold, CPU cycles, flash bytes, peak RAM, cache misses (where available) and energy per query if measured.
- Compare table load/initialization time separately from steady state; repeat on at least one constrained MCU and native/WASM.

## Decision

Promote only a table with certified output, documented domain, and measured advantage in a named deployment envelope. Prefer arithmetic outside the codebook or where table cache/ROM costs dominate. If the LUT performs worse than direct compute, publish the negative result.
