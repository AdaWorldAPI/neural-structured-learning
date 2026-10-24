# H6 / DR³ hypercube: reversible addressing and adversarial sparsity

**Status:** proposed address/layout experiment; not proof that a hypercube engine exists. Companion to [PR #6](https://github.com/AdaWorldAPI/neural-structured-learning/pull/6).

## What a cube may mean

Hypothesis: a 3-axis relation index can preserve **typed identity and dependency semantics** while traversing only occupied/focused subregions.

Candidate domains are **alternative views**, not a single implied universal schema:
- U × U × R: source identity, target identity, relation type.
- U × P × T: identity, property/facet, logical time bucket.
- Operator × InputPort × OutputPort: computational graph dependency.
- DirectionA × DirectionB × RuleFamily: quantized pair operation by context.

No rule says "axis 3 means time" across all workloads. Every view must carry a typed domain signature and an invertible ordinal dictionary. A rank-3 bitcube is not an arbitrary-rank dense tensor value store.

## Exact capacity and bit addressing

DR64³: 64³ = 262,144 logical Boolean cells = 32 KiB dense bit storage.
DR256³: 256³ = 16,777,216 cells = 2 MiB dense bit storage.
A 256×256 plane is 65,536 cells = 8 KiB of bits; 256 such planes are 2 MiB, before headers/metadata/values.

Candidate contiguous flat index:
  index(x,y,z) = (x * Y + y) * Z + z  for x∈[0,X), y∈[0,Y), z∈[0,Z).

Candidate equal-width Morton index:
  morton3(x,y,z) = Σ_i ((x_i << (3i)) | (y_i << (3i+1)) | (z_i << (3i+2))).
For 64³ use 6 bits/axis -> 18-bit index. For 256³ use 8 bits/axis -> 24-bit index. The bit interleave should be independently inverted; do **not** assume 2D HHTL Morton routines already implement 3D ordering.

Prove that decode(encode(x,y,z)) = (x,y,z) for every admitted coordinate. For DR64³ an exhaustive 262,144-coordinate test is straightforward. For DR256³ sample plus boundary/permutation fixtures initially, with an exhaustive test only if runtime budget permits.

Define bit index, byte order and publication/revision digest; a wrong endian convention can appear correct in uniform-mask tests.

## Data cannot disappear behind an address

- Presence Boolean vs null/missing vs typed value plane is explicitly represented.
- A 1-bit presence cube with 16,777,216 cells is 2 MiB; an f32 value at every populated cell would add up to 64 MiB for a fully dense 256³ cube.
- Value SoA, sparse value ordinals, compression, tile metadata, source dictionaries and stale-frame checks have separate measured storage costs.
- A relation code and neural activation may occupy the same addressing abstraction, but their algebraic semantics are different. No "neuron = fact" shortcut.

## Candidate focus/paging variants

A. Dense bitcube with direct bit index.
B. 2D DR256² planes, sparse third-axis directory.
C. 16³ tiles with per-tile occupancy mask and sparse directory; 16³ Boolean tile is 512 bytes.
D. CSR/COO adjacency + SoA payload reference baseline, with optional roaring-like bitset or partitioned sorted vectors.
E. #1762-inspired materialized focus lease: full SourceId + MapId + Scope, with own invalidation and dirty-flush behavior explicitly *not yet verified*.

For a view with only 1 out of 256 third-axis slices occupied, dense 2 MiB may be very poor value compared with sparse 8 KiB plane plus directory. In a dense hot neighborhood, direct indexing may win.

## Preregistered workload matrix

| ID | Shape | Expected failure surface |
| --- | --- | --- |
| C01 | completely dense, read-many | directory overhead vs dense bit arithmetic |
| C02 | 0.01% uniform random occupancy | dense memory blowup |
| C03 | 1% occupancy in 1/16 tiles | measured focus benefit vs tile lookup cost |
| C04 | adversarial diagonal-only relations | sparse CSR/COO likely better |
| C05 | power-law hub degree | uneven tile utilization and cache churn |
| C06 | 90% relations on one hot plane | plane directory may beat 3D tiling |
| C07 | frequent edits to one temporal plane | invalidation / stale lease costs |
| C08 | same ordinal reused with changed source seal | must refuse |
| C09 | two equal data populations, differing axis permutations | locality must be measured for each axis order |
| C10 | all boundary coordinates and tile crossings | zero collisions and exact inverse map |

## Independent correctness and cost oracle

Use a plain typed triple set plus sorted map as semantic oracle. Compare insert/delete/contains, enumeration, range/focus query, union, intersection and relation composition only where those operations are truly defined for the typed domains. Do not claim a generic "tensor MatMul" follows from Boolean relation composition.

Metric columns: logical cells, nonzero occupancy, dense/SoA/directory bytes, resident bytes, page faults, p50/p95/p99 latency, update invalidation, cold/warm focus, CPU cache misses, throughput.

## Go/no-go

Any coordinate collision, silent alias, invalid seal accepted, or untracked distinction between absent and zero is a correctness failure. If cost loses to CSR across practical sparsities, reject that layout for those workloads. **The earlier 1/16 focus ratio is a workload hypothesis, not a storage guarantee or theorem.**
