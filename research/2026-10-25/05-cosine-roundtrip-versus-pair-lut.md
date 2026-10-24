# Pairwise LUT vs cosine roundtrip: measure before replacing

Status: RESEARCH PROPOSAL, unimplemented. The claim is local to a proposed *discrete geometric operator*, not a criticism of all of TensorFlow Lite.

## Source anchor
NSL `tools/build_graph.py` normalizes embeddings and uses cosine similarity for graph edges. `lib/distances.py` exposes cosine and other metrics. This establishes a measurable cosine-based baseline, but does not establish that NSL necessarily uses an `acos(cos(x))` roundtrip. Such a roundtrip must be found in a specific algorithm before asserting it exists.

## Hypothesis
For a finite index pair (a,b), tabulate a well-defined geometric relation or composition once, then access L[a,b] directly. For 256x256 cells: 65,536 entries = 64 KiB u8, 128 KiB u16, 256 KiB f32, before metadata. A factored or focus-local table may reduce active memory, at the cost of additional addressing and misses.

## Design experiment
1. Pin whether a,b mean quantized directions, angular bins, relation endpoints, or learned codebook entries. No ambiguous LUT semantics.
2. Specify ideal mathematical oracle f(a,b), quantizer Q, rounding mode, NaN/sentinel policy, angle topology and error metric (including circular distance).
3. Generate reference table using high precision offline; certify code selection at bin boundaries, report undecidable cases as generator failure.
4. Implement three benchmark paths: direct trigonometry, full pair-LUT, and factorized/focused LUT; compare appropriate real baseline, not an invented roundtrip.
5. Report flash, RAM, cache misses, instruction count and latency on native, WASM and a concrete Arduino-class target. For constrained boards, 64 KiB ROM may be too large.
6. Test chained compositions for accumulated angular drift and algebraic invariants, not just one-off lookup error.

## Go / no-go
- No worse domain-specific angular/relational error than specified bound.
- Zero uncertified entries; signed zero, branch cut, periodicity, poles and invalid codes covered.
- Baseline runtime compared under identical I/O assumptions, including table-generation cost amortized separately.
- No speedup claim until measurements; reject table strategy if cache/flash constraints erase gain.

## Connections
ndarray #370: Codec8 calibrations and compile-time policy. #375: TFLM requantization exactness. lance-graph #1762: focused gathering mechanism is *potentially reusable* but unrelated to LUT certification until proven.

References: https://github.com/AdaWorldAPI/ndarray/pull/370 ; https://github.com/AdaWorldAPI/ndarray/pull/375 ; https://github.com/AdaWorldAPI/lance-graph/pull/1762
