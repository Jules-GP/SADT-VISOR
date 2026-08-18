# sadt-areg-ios

Registers a follow-up intraoral scan onto its baseline, by ICP on a patch of the
arch that does not move with growth or treatment — the palate, or the band
around the mucogingival line.

Split out of the former single `AREG`; see `../common/README.md` for what the
two engines still share and why.

## The ICP is not deterministic, and here is its measured spread

Registering the same pair twice, with the same code and the same weights,
does not give the same mesh. Measured on upstream's own `AREG_test_scans`
(`A2_UpperT1.vtk` / `A2_UpperT2.vtk`, 75 867 points, 57.5 mm across), on the
registered T2 — the mesh the ICP actually moves:

| comparison | mean | p95 | max | identical points |
|---|---|---|---|---|
| direct vs direct | 0.1879 mm | 0.3376 mm | 0.3797 mm | 0 / 75 867 |
| direct vs HTTP | 0.1172 mm | 0.2086 mm | 0.2605 mm | 0 / 75 867 |
| direct2 vs HTTP | 0.1896 mm | 0.3236 mm | 0.3614 mm | 0 / 75 867 |

**Two direct runs differ from each other MORE than a direct run differs from
the same job dispatched over HTTP.** That is the number that matters: it says
the spread belongs to the ICP, not to the server. Had the dispatch, the
virtualenv or the environment perturbed convergence, direct-vs-direct would sit
near zero and the other two would not.

So the reference for this tool is **not** zero. It is ~0.19 mm mean and ~0.38 mm
max on this pair, about 0.7 % of the mesh's own extent, and a comparison should
be read against that rather than against bit-equality. The registered T1 IS
bit-identical, being the mesh nothing moves.

Note what this is not: the CBCT engine registers with elastix on the CPU and is
bit-exact, volume and `.tfm` alike. The difference is the algorithm, not the
plumbing.

**Not measured here:** whether the spread is the same on other pairs, or whether
it grows on a harder registration. One pair is enough to establish that a
tolerance is needed and roughly where it sits; it is not enough to publish a
bound. Anything clinical should re-measure on its own data.
