# DR³ addressing: a falsifiable hypercube rather than an assumed tensor engine

Status: RESEARCH PROPOSAL, unimplemented. DR64³ / DR256³ are architectural hypotheses.

## Hypothesis
A 3-axis relational address space may host identity, relation/property and time/channel/model dependencies in pages, while active focus traverses only relevant tiles. Different datasets may require different axis semantics; rank-3 address space is NOT equivalent to arbitrary tensor rank.

## Explicit arithmetic
- 64^3 = 262,144 logical cells; at 1 bit/cell = 32 KiB dense. 256^3 = 16,777,216 cells = 2 MiB dense bitmask.
- A DR256² plane has 65,536 cells = 8 KiB at 1 bit/cell; 256 planes require 2 MiB. Attribute values, sparse index, pointer overhead and cache structures are additional.
- Earlier DR focus work (#1762) stores gathered i32 facet columns. It does NOT implement a DR³ cube.

## Probe matrix
A. Node x node x relation-type planes on tiny typed graph.
B. Node x feature x temporal step on synthetic event stream.
C. Neural dependency indices x operator channel x quantized parameter family from a fixed tiny model.
D. Deliberately adversarial skewed/sparse graph, to test whether 3D tiling is worse than CSR/COO/SoA.

## Method
1. Specify canonical address and reversibility, Morton/HHTL mapping only if proven bijective for populated range.
2. Define per-axis ID mapping, tile bounds, eviction and branch/index seals; forbid reuse across incompatible scope.
3. Compare query correctness against a plain adjacency-list + sparse SoA oracle for each matrix, including missing and duplicate cells.
4. Benchmark memory footprint, cache/TLB misses, load time, active focus fraction, cold/warm traversal and update cost.
5. Perturb one tile; measure actual invalidation footprint and correctness under concurrent readers.

## Falsifiers
- Any aliasing of logical coordinates or silent overflow is rejection.
- Report worst-case sparse memory blowup; no 1/16 focus benefit assumption.
- Compare against optimized CSR and dense bitmap baselines, not only a naive array.
- Independent verification of layout; no claim of general tensor replacement without rank-4+/broadcast/stride/quantization semantics covered.

## Dependencies
lance-graph #1762 focus lease and seals; ndarray #372-375 low-level SIMD/quantized kernels may be reused but do not validate layout.

References: https://github.com/AdaWorldAPI/lance-graph/pull/1762 ; https://github.com/AdaWorldAPI/ndarray/pull/375
