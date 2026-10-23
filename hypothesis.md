# Hypotheses

Each hypothesis is committed before the experiment it describes is run.

## Part 3: Compression (SIFT, 1M vectors)

**Setup.** IVF-PQ with nlist = 2048 (as in Part 2) and 8-bit codes, varying the
number of PQ sub-vectors m in {8, 16, 32, 64} (8 to 64 bytes per vector,
16 to 2 dimensions per sub-vector). Each index is searched two ways while
sweeping nprobe:

1. plain IVF-PQ: the top 10 by approximate (PQ) distance;
2. IVF-PQ + re-ranking: take the top 100 by PQ distance, compute their exact
   distances from the original float vectors, and return the best 10.

The control is IVF-Flat with the same nlist (no compression), from Part 2.
Only m and the re-ranking switch change.

**Hypothesis.**

- *Plain IVF-PQ has a recall ceiling set by m, not by nprobe.* PQ distances
  are approximate; when two candidates' true distances differ by less than the
  quantisation error, PQ can order them wrongly, and probing more lists
  cannot fix that. Fewer sub-vectors means coarser codes and a lower ceiling.
  We expect the ceiling to rise steadily with m and to stay below 0.95 even
  at m = 64 (Part 2 already showed m = 64 levelling off below 0.90).
- *Re-ranking the top 100 removes most of that ceiling.* PQ is usually good
  enough to put the true top 10 somewhere in its top 100, even when it orders
  them wrongly, and exact re-ranking then fixes the order. We expect re-ranked
  recall to reach >= 0.95 for m >= 16, and to be limited only by nprobe
  (i.e. approach the IVF-Flat curve). For m = 8 the codes are so coarse that
  some true neighbours will fall outside the top 100, so re-ranking will help
  a lot but may still fall short of 0.95.
- *Re-ranking costs little time but most of the memory saving.* 100 exact
  distances (100 x 128 floats) per query is small next to scanning thousands
  of PQ codes, so QPS at a given nprobe should drop by well under 2x. But
  re-ranking needs the original vectors (512 B/vector) kept in memory, so the
  index goes from m bytes/vector back to about 512 + m bytes/vector.
- *Consequence.* At recall 0.95, IVF-PQ + re-ranking with a small m should be
  at least as fast as IVF-Flat, because scanning m-byte codes is cheaper than
  scanning 512-byte vectors; plain IVF-PQ cannot reach 0.95 at all.

## Part 4: Hard geometry (synthetic data)

**Question.** How does the *intrinsic* dimension of the data change the work
HNSW and IVF-Flat need to reach a given recall?

**Data.** 200,000 base vectors and 10,000 queries drawn from a standard
Gaussian in d dimensions, d in {8, 16, 32, 64, 128}, then mapped into 128
dimensions with a fixed random orthonormal matrix. Every setting is stored as
128-d float vectors, so one distance computation costs the same in all of
them; only the geometry changes. Seeded, so every run makes the same data.
**Control:** the first 200,000 SIFT vectors (128-d, real data), unchanged,
run through the same pipeline and the same index settings.

**Indexes (fixed build settings).** HNSW with M = 16, efConstruction = 100,
sweeping efSearch; IVF-Flat with nlist = 512, sweeping nprobe. Same recall
and timing method as Part 2 (tune on the first 5,000 queries, report on the
last 5,000). We also record the relative contrast (mean distance to a
random base vector divided by the distance to the nearest neighbour) and the
number of distance computations per query.

**Hypothesis.**

- *Contrast collapses as d grows.* With more independent directions, all
  distances concentrate around the same value, so the nearest neighbour is
  barely closer than a random point. We expect relative contrast to fall
  steadily with d and approach 1 at d = 128.
- *IVF-Flat degrades the most.* With low contrast a query's true neighbours
  are spread over many Voronoi cells, since most points lie near a cell
  boundary in high dimensions. We expect the fraction of lists IVF must probe
  for recall 0.95 to grow from a few percent at d = 8 to more than half of
  all lists at d = 128, so its QPS at 0.95 falls close to exact search.
- *HNSW also degrades, but more slowly.* Greedy graph search relies on
  "closer neighbours lead towards the target"; with low contrast that signal
  is weak, so efSearch and distance computations per query must rise. We
  expect distance computations at recall 0.95 to grow by more than 10x from
  d = 8 to d = 128, and HNSW may not reach 0.95 at all at d = 128 within
  efSearch <= 1024. HNSW should still beat IVF-Flat at every d.
- *Because the per-distance cost is fixed, the QPS drop should track the
  growth in distance computations.*
- *Real SIFT behaves like low-dimensional data.* Although SIFT has 128
  dimensions, its points lie near a much lower-dimensional structure, so we
  expect its contrast and the work needed to sit between the d = 8 and
  d = 32 synthetic settings, far from d = 128. This would mean the Part 2
  headline numbers for SIFT are much better than what the same indexes would
  give on truly high-dimensional data.

**Not tested here.** Changing the stored (ambient) dimension, clustered or
skewed data, dataset size, and compressed indexes.
