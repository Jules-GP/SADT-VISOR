# AutoMatrix

Apply a transform to scans, segmentations and landmark files.

Each patient's transform is matched to their files by name, then applied:
SimpleITK resamples an image, and a landmark file has its points moved by the
INVERSE — a transform produced by a registration maps image space, and a point
follows the other way.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, `AutoMatrix/` and
`Automatrix_CLI/`. Of its 1 502 lines, **1 239 are the Qt panel**; the 263 that
compute are what this package holds.

## Three defects, and what they cost

### The branch that reads AREG's output could never run

```python
matrix_path = os.path.join(
    args.matrix_lineEdit,      # <- line 265
    subdir, f"{patient_id}_OutReg", f"{patient_id}_{matrix_filename}")
```

`matrix_lineEdit` is not among the nine arguments argparse declares. The whole
`fromAreg == "True"` path — the one that consumes what AREG wrote — raises
`AttributeError` on its first landmark file. It is the integration between the
two tools, and it has never executed.

### The jaws were swapped

```python
"_L": ("Maxilla",  "MAXReg_matrix.tfm"),     # _L is Lower  -> mandible
"_U": ("Mandible", "MANDReg_matrix.tfm"),    # _U is Upper  -> maxilla
```

A lower-arch file would be transformed by the **maxillary** registration and
written out as a success — the family of defect ASO had, where a maxillary mesh
was registered against the mandibular reference and reported as fine. The first
defect hid this one: the branch crashed before anyone saw it.

The mapping is not restated here at all. A transform is matched to a file by
patient, and a patient's several transforms are each applied, told apart by
`name_output_after_transform`.

### The patient came from fifteen chained splits

```python
file_pat = os.path.basename(file).split('_Seg')[0].split('_seg')[0].split('_Scan')[0] \
    ... .split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('_MR')[0].split('.')[0]
for i in range(50):
    file_pat = file_pat.split('_T' + str(i))[0]
```

A patient genuinely named `P_Seg1` is truncated to `P`, and the trailing
`.split('.')[0]` breaks any name holding a dot. Suffix matching was also a
substring test (`if suffix in os.path.basename(scan)`), so `_L` matched anywhere
in the name and the dict's iteration order decided which won.

This tool uses `sadt_areg_common.pairing.patient_stem` — CONTRIBUTING.md lists
identity derivation as one of the three things to share rather than copy —
with the transform vocabulary the shared table does not know (`transform`,
`matrix`, `warp`, `reg`, and the region and jaw tokens). Without it,
`C_0001_CB_Reg_transform.tfm` keys to `C_0001_CB_Reg_transform` and matches no
scan, which is this tool's whole job.

## Kept, because it is the rule that matters

Nearest neighbour for a segmentation, linear otherwise. Interpolating a label
map linearly produces labels that were never in it: a test shifts a volume
holding labels 1 and 3 by a third of a voxel and asserts that **2 appears under
linear interpolation and does not under nearest neighbour**.

## Not ported

The `<filter-progress>` prints and their `time.sleep(0.2)`; the mirror-transform
special case, which forced the image as its own reference on any matrix whose
name contained "mirror" — a substring test on a file name deciding a resampling
grid.

## Reporting

`AutoMatrix_report.json` names `without_a_transform` and
`transforms_without_a_file`. Upstream skipped both in silence, so a run could
transform 3 of 40 patients and look complete.

## Working on it

```bash
cd tools/AutoMatrix
uv sync
uv run pytest -m "not gpu"
.venv/bin/python ../../scripts/describe.py .
```

## Data

The manifest declares 1.4 GB of models for this tool, none of which this port
needs: it has no network. The loopback run recorded in the pull request used a
transform produced by GreedyReg and the CBCT it was computed from.
