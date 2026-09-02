# Test data

Nothing is committed here. The suite builds its own 16³ synthetic CBCT volumes
with SimpleITK, its own labelled meshes with VTK, and its own landmark sets --
seven points in a plausible skull-ish arrangement, no three collinear, so the
coarse alignment can always find a usable triplet. A known rotation is applied
and the registration has to recover it.

That covers 80 of the 81 tests, including six that run the tool out of process
in its own venv. Only the ALI chain needs real files.

## Staging what the chain test needs

`test_fully_automated_cbct_drives_ali_through_the_supervisor` is marked `gpu`
and `models`. It runs the **real** ALI through the supervisor, so it needs that
tool built and its CBCT bundle staged:

```bash
cd ../ALI && uv sync          # ~8.8 GB, CUDA wheels
scripts/setup-models.sh --tool ALI   # from a VISOR-serve checkout
```

| File | Layout | Size |
|---|---|---|
| `ALI_CBCT_Models` | `<landmark>/<scale>/*.pth`, scales `1` and `0-3` | 4.7 GB |

The chain test is pointed at its data by environment variable, and uses a **real
scan**: the synthetic 16³ volume the rest of the suite builds is not a head, and
the landmark agent converges on nothing in it -- a fact about the fixture, not
about the chain.

```bash
SADT_ALI_SCAN=/path/to/DATA/ALI/testfiles/MG_test_scan.nii.gz \
SADT_ALI_CBCT_MODELS=/path/to/DATA/ALI/models/ALI_CBCT_Models \
uv run pytest -m "gpu and models"
```

It skips rather than fails when either is unset, or when `tools/ALI` has no
venv. It drives ASO through `scripts/run_tool.py` rather than
`sadt_testkit.run_tool`, because only the former supplies a supervisor -- the
latter is the server runner's contract, which does not.

Only the seven landmarks in `DEFAULT_CBCT_LANDMARKS` are asked for, so a partial
bundle carrying `Ba`, `S`, `N`, `RPo`, `LPo`, `ROr` and `LOr` at both scales is
enough -- about 300 MB rather than 4.7 GB.

## Reference bundles

A CBCT *reference* is one already-oriented case's landmark file; it is data, not
weights, and the reference scan is not read. The two published bundles carry
**disjoint** landmark sets:

| Bundle | Landmarks |
|---|---|
| `CBCT_Gold_Frankfurt_Horizontal_Midsagittal_Plane` | Ba, S, N, RPo, LPo, ROr, LOr |
| Occlusal + Midsagittal | ANS, IF, PNS, UL6O, UR1O, UR6O |

`run()`'s defaults are the first set. Choosing the second without changing
`cbct_landmarks` is refused up front with a message naming what the reference
actually offers -- the failure that used to be forty identical per-patient ones.

Write manual runs into the repository's gitignored `output/` directory, never
next to the inputs.
