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
