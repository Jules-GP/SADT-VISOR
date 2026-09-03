# AutoCrop3D

Crop CBCT volumes and their segmentations to a Region Of Interest drawn in
Slicer.

The ROI is a `.mrk.json` markup: its centre and size give two physical corners,
those corners map to voxel indices, and the volume is sliced between them.
`keep_original_size` puts the crop back into a volume of the original geometry
with everything outside the box zeroed, so the result still overlays the scan it
came from. A cropped label map can also be turned into a smoothed `.vtk`
surface.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, path
`AutoCrop3D/Crop_Volumes_CLI/`, commit `b1ae78c` (2026-05-26).
Upstream pins kept as-is: none — upstream pins nothing. SimpleITK 2.5.6,
numpy 2.3.2 and vtk 9.6.2 are this repository's shared pins.
Changes from upstream: thirteen defects, listed below, plus two additions
(the ROI's declared coordinate system is now read, and the surface is written
in the patient space it belongs to).

Of the 1 457 lines upstream, **1 136 are the Qt panel**. The 321 that compute
are what this package holds — minus `Crop_Volumes_utils/CropCBCT.py`, 62 lines
its own header marks `UNUSED`.

## What it does

| | |
|---|---|
| Inputs | `scans`: a volume or a folder of them (`.nii`, `.nii.gz`, `.nrrd`, `.gipl`, `.gipl.gz`), searched recursively. `roi`: a `.mrk.json` ROI or a folder of them. `output_dir`: where results go. |
| Options | `suffix`, `keep_original_size`, `surfaces` (segmentations/all/none), `surface_padding_mm`, `surface_smoothing_iterations`. |
| Outputs | One cropped volume per scan, mirroring the input tree, plus `<name>_<suffix>_vtk.vtk` for each surfaced label map and `AutoCrop3D_report.json`. |
| Model files | **None.** A crop is arithmetic on an image grid: no weights, no GPU, no network. The empty `DATA/AutoCrop3D/` is correct. |

## The interface

```python
def run(
    scans: Path,
    roi: Path,
    output_dir: Path,
    suffix: str = "cropped",
    keep_original_size: bool = False,
    surfaces: Literal["segmentations", "all", "none"] = "segmentations",
    surface_padding_mm: float = 5.0,
    surface_smoothing_iterations: int = 5,
) -> Path:
```

Upstream's six parameters map onto it as: `scan_files_path` → `scans`,
`path_ROI_file` → `roi`, `output_path` → `output_dir`, `suffix` → `suffix`,
`box_Size` → `keep_original_size`. `logPath` is **not ported**: it is the file
the Qt progress bar polled for an integer, and this side reports through the
run report and the server's own progress channel.

## The identifier, which is the whole defect

Upstream derived a patient key twice, with two rules that could not agree:

```python
# AutoCrop3D_CLI.py:59  -- the SCAN side, fifteen chained splits
patient = os.path.basename(p).split('_Scan')[0].split('_scan')[0]. ... .split('.')[0]

# Crop_Volumes_utils/FilesType.py:78  -- the ROI side, the first token only
patient = os.path.basename(file).split('_')[0]
```

So `PatientA_01_Scan.nii.gz` keyed as `PatientA_01` while its own
`PatientA_01_ROI.mrk.json` keyed as `PatientA`. The lookup missed, a bare
`except` logged and `continue`d, and **every cohort whose identifiers contain
an underscore was skipped in full — the CLI exited 0 with an empty output
folder.** Reproduced here on the real 512×512×365 CBCT in `DATA/`:

```console
$ python AutoCrop3D_CLI.py cohort/scans cohort/rois out cropped False log.txt
AutoCrop3D_CLI - WARNING - No ROI for patient:Patient_02
AutoCrop3D_CLI - WARNING - No ROI for patient:Patient_01
$ echo $?; ls out | wc -l
0
0
```

`pipeline.patient_key` is one function, run over both sides, so their equality
is structural rather than a coincidence of two rules. It is built on
`sadt_areg_common.pairing` for the two things CONTRIBUTING.md says must not be
re-derived per tool — what a compound scan extension is
(`split_scan_extension`), and how a stem breaks into tokens (`split_parts`).

**It deliberately does not call `pairing.patient_stem`.** That function strips a
previous run's suffixes with `stem.find(suffix)`, a substring match, which
reproduces the very truncation listed as defect 2 below: `SMITH_ORTHO` contains
`_OR`, so `patient_stem` returns `SMITH`. Here, where the key is what pairs a
scan with its box, that turns two subjects into one silently. `patient_key`
drops the same vocabulary as **whole tokens** instead, and never drops the
leading one — a patient really can be called `MAX_01`.

## The thirteen defects

Each is fixed, each has a named test, and the ones marked ✱ were reproduced
against upstream on real data before being fixed.

| # | Upstream | What it cost | Now |
|---|---|---|---|
| 1 ✱ | Two identifier rules (`CLI:59` vs `FilesType:78`) | Any cohort with an underscore in its IDs: exit 0, empty output | One `patient_key` on both sides |
| 2 | `.split('_OR')[0]`, `.split('_T1')[0]`, `.split('.')[0]` | `SMITH_ORTHO`→`SMITH`; T1 and T2 became one subject; `Patient.01`→`Patient` | Whole-token drops, timepoint kept, compound extension split off |
| 3 | `result[patient] = file`, no collision check | `P01_T1_ROI` and `P01_T2_ROI` both keyed `P01`; **T1 cropped with T2's box, looking plausible** | An undecidable pairing is a `ValueError` naming both files |
| 4 ✱ | ROI table built only `if len(ROIList) > 1` | A folder holding exactly one ROI: `IsADirectoryError` on the first patient | One file, or a folder holding one, crops every scan |
| 5 | `.nrrd.gz` discovered; read outside the `try`, write inside a bare `except:` | One such file ended the batch; or produced nothing while the counter advanced | Guarded per file, named in the report. Plain `.nii`/`.nrrd`/`.gipl` folders are accepted too — the *panel* refused them |
| 6 ✱ | `os.path.join(out, rel).replace(basename(rel), filename)` | In single-file mode `rel` is `"."`, so the file name replaced **every dot in the path**, and `os.makedirs` built the result | `relpath` + `Path` joins |
| 7 | Bare `except:` around the write, then `index += 1` regardless | "Scan(s) cropped with success" for a run that wrote nothing | Per-scan report; a run that wrote nothing raises |
| 8 ✱ | `"image_padded.nii.gz"`, a fixed RELATIVE path, removed only on success | Two concurrent requests overwrote each other's temp volume; a failure left patient anatomy in the CWD | One `TemporaryDirectory` per run, removed in a `finally` |
| 9 ✱ | `LABEL_COLORS[np.max(img_arr)]`, table keyed 1..6 | `KeyError` above 6 labels and for an empty crop, swallowed by `except: pass`; and **every cell got the maximum label's colour**, so the per-label colouring never happened | Colours per label, from `present_labels` (computed upstream at `:29-32` and never used); a generated palette beyond 6 |
| 10 | `if "seg" in ScanOutPath.lower()` | An output folder `/data/Segmentations/` surfaced every CBCT; `Mandible.nii.gz` got none | Explicit `surfaces` argument; the token test is on the file's own stem |
| 11 | `if originalSize == 'True'` | `"true"`, `"1"` and a real `True` all took the else branch | A real `bool` |
| 12 | `padding_size = [50, 50, 10]` VOXELS; `GenerateValues(100, 1, 100)` | The margin meant 25 mm on a 0.5 mm scan and 8 mm on a 0.16 mm one; nothing above label 100 was ever contoured | `surface_padding_mm`, converted per axis; one contour per label present |
| 13 | `sitk.ReadImage` and `json.load(open(...))` unguarded in the loop | One bad file ended the batch | Guarded per item, reported per item |

Defect 6, reproduced: given an output directory `…/study.v2/out`, upstream
created

```
…/studyMG_test_scan_cropped.nii.gzv2/out/MG_test_scan_cropped.nii.gz
```

Defect 9, reproduced on a two-label map: upstream wrote every cell
`(230, 220, 70)` — `LABEL_COLORS[4]`, the maximum label — and on a seven-label
map raised `KeyError np.int16(7)`, leaving `image_padded.nii.gz` in the working
directory.

## Two additions, beyond the list

**The ROI's coordinate system is read.** Slicer writes `coordinateSystem` into
a markups file and upstream never looked at it, so an ROI saved in RAS was
applied to an LPS volume and cropped the mirror image of what was drawn. LPS is
the default when the field is absent, which is what upstream implicitly assumed
and what its own shipped test ROI relies on. An unknown value is a `ValueError`.

**The surface is written in patient space.** `vtkNIFTIImageReader` deliberately
does not apply a file's qform: it reports origin `(0, 0, 0)` and leaves the
matrix to the caller. Upstream never applied it, so every `.vtk` it produced was
offset by the volume's origin, un-rotated, and shifted again by the padding.
Measured on a label map at origin `(5, -3, 2)`: upstream's surface came out at
`x ∈ [21.1, 26.1]` for anatomy sitting at `x ∈ [6.0, 11.2]` — 15 mm away, in
every axis. Inside the Slicer module nobody saw it, because the panel loads the
segmentation rather than the `.vtk`; the moment the returned file is opened
beside its scan, the mesh floats away from the anatomy. The same shape of defect
as ALI's `display.visibility: false`. One 4×4 matrix, no resampling.

## Not ported, deliberately

**The second cropping engine.** `Crop_Volumes_UI/AutoCrop3D.py:915-998`
(`processCropVolume`, the "Crop Volume for tilted images" checkbox) is a
different implementation built on `slicer.modules.cropvolume.logic()` — a
Slicer *loadable* module, not a CLI, reached through MRML nodes and a
`vtkMRMLCropVolumeParametersNode`. It has **no headless equivalent**: there is
no way to call it without a running Slicer application, so it cannot be a
subprocess in a tool venv. It is the only path that honours a rotated ROI, by
resampling the volume onto the box's own axes.

Consequently, and as upstream's CLI already did, **ROI orientation is ignored**:
`AutoCrop3D_CLI.py:70-72` reads `center` and `size` and nothing else, so a
rotated ROI is cropped as its axis-aligned self. That is not silent any more —
`roi_orientation_ignored` appears in the run report — and the flag fires only
for a *genuine* rotation: a signed axis permutation (upstream's own test ROI
carries a 180° flip about z) maps an axis-aligned box onto itself, so ignoring
it is exact. An oriented crop is a resampling and belongs in a tool that
resamples.

**`Crop_Volumes_utils/CropCBCT.py`** (62 lines), whose own header says
`UNUSED`, and which nothing imports but `__init__.py`.

**`logPath`.** The file the Qt progress bar polled for an integer.

**The smoothing feature angle (120°) and relaxation factor (0.6)** stay module
constants rather than arguments. They are properties of the smoothing recipe,
not per-request clinical choices; the iteration count, which someone genuinely
turns down when a thin structure loses detail, *is* an argument. `pipeline.py`
names both, so changing them is one edit rather than a hunt.

**The panel's own extension filter.** `AutoCrop3D.py:808` searched only
`.nii.gz`, `.nrrd.gz` and `.gipl.gz` and refused a folder of plain `.nii` /
`.nrrd` / `.gipl` that the CLI behind it reads perfectly well. Discovery here
follows the CLI (`:48`), which is the wider and correct list.

## Working on it

```bash
cd tools/AutoCrop3D
uv sync --all-groups
uv run pytest                          # 102 tests, ~1s, no weights and no GPU
.venv/bin/python ../../scripts/describe.py .
```

## Validated against

**`DATA/AMASSS/testfiles/MG_test_scan.nii.gz`** — a real 512×512×365 CBCT at
0.33 mm — cropped with **upstream's own test ROI**
(`AutoCrop3D/Crop_Volumes_UI/Testing/Test_data/ROI.mrk.zip`, centre
`(-0.005, 0.902, -2.874)`, size `(30.23, 22.60, 30.05)`), run through the
upstream CLI and through this package and compared:

| | upstream | port |
|---|---|---|
| size | (91, 68, 91) | (91, 68, 91) |
| origin | (-15.180000305, -10.560000420, -18.149768829) | identical |
| spacing, direction, pixel type | | identical |
| voxels | | **byte-identical, max abs diff 0** |

Identical again with `keep_original_size=True` (512×512×365 out, 562 731
non-zero voxels, same geometry). A crop is pure geometry, so exact agreement is
the right bar and it is met.

The surface path is **deliberately not** byte-identical: it is coloured per
label rather than uniformly, contoured only on the labels present, and placed in
patient space. Those three differences are the fixes above, each with a test.

Everything else runs against volumes and `.mrk.json` files built in
`tests/test_run.py` with SimpleITK — voxel values encode their own index, so a
crop that lost track of its bounds cannot produce the right numbers by accident.
