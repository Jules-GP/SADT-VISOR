# CLIC

Segment the impacted canine on a CBCT scan.

A torchvision Mask R-CNN (`maskrcnn_resnet50_fpn`) is applied to each axial
slice of the volume as a 3-channel image. Detections that clear a score
threshold are painted into a label volume that keeps the input's affine and
header, so the scan and its segmentation open aligned in a viewer.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, `CLIC/`, whose module is
`CLIC.py` (the panel) and `runner/clic_runner.py` (the work). Of its 732 lines,
622 are the Qt panel and the conda environment it installs on the user's
machine; the 110 that compute are what this package holds.

## Changes from upstream

- **The checkpoint is named, not guessed.** Upstream took
  `sorted(model_folder.glob("*.pth"))[0]` -- the alphabetically first file, with
  no check that it was the only one, and an `IndexError` on an empty folder.
  Which model vintage ran must never depend on file names.
- **The class count is read from the checkpoint.** Upstream built both heads for
  4 classes whatever the file held, so a checkpoint trained on a different count
  died inside `load_state_dict` on a shape mismatch. The count is in the
  weights: the box predictor's `cls_score` bias has one entry per class.
- **A folder is walked recursively, and files are returned.** Upstream's
  `_collect_scans` returned SUBDIRECTORIES whenever any of them held a scan, and
  the runner then called `nib.load` on a directory. It also advertised `.nrrd`,
  `.mha` and `.mhd`, none of which nibabel reads -- accepted and then ignored,
  which is the trap `.stl` fell into in ALI. This tool advertises `.nii` and
  `.nii.gz`, which is what it reads.
- **One scan failing costs one scan.** Upstream ran one process per scan from
  the panel, so it never had to say this; the loop is inside the tool now, and a
  failure is recorded per scan.
- **The output mirrors the input tree.** A cohort is exported one folder per
  patient and every scan inside is called the same thing, so naming the output
  after the base name alone wrote both of them to `scan_seg.nii.gz`: one
  patient's segmentation silently replaced another's while the report said both
  had been segmented. Scans are keyed by their path relative to the input root,
  and that is the path the report carries. A single file named directly is its
  own root and lands straight in `output_dir`.
- **`score_threshold` is an argument.** Hardcoded to 0.7 upstream. It is the one
  knob that moves the segmentation, so a caller comparing two runs has to be
  able to say which value produced which -- and it is recorded in the report.
- **Not ported**: the `[PROGRESS]`/`[LOG]`/`[SEG]` stdout protocol, which is how
  the panel drove its progress bar, and the runtime model download.

## Reporting

`CLIC_report.json` carries the model, the class count, the device, the
threshold, and per scan: its path relative to the input root, the slice count,
how many detections cleared the threshold, the labels actually present in the
output, and a note when nothing cleared it. That last one matters: an empty segmentation is a legitimate answer
AND the signature of a wrong checkpoint, and only the caller can tell them
apart.

## Working on it

```bash
cd tools/CLIC
uv sync
uv run pytest                                    # 127 tests, no GPU and no checkpoint
.venv/bin/python ../../scripts/describe.py .     # the schema the server publishes
```

The `gpu` and `models` markers are **deselected** by the default run rather
than skipped: a suite whose green line is half skips says nothing. The two
`models` tests are the only claims a stub cannot make -- that the real 176 MB
checkpoint loads into the heads the class count read out of it, and that the
real detector's output dict is the one `segment_volume` indexes. Run them by
hand and report the result in the pull request:

```bash
SADT_CLIC_MODEL=../../../VISOR-serve/DATA/CLIC/models/final_model.pth \
    uv run pytest -m models -o addopts=
```

## Data

`DATA/CLIC/models/final_model.pth` (176 MB), staged by
`scripts/setup-models.sh --tool CLIC` from the manifest.

**No test scan is staged.** The manifest declares no `testfiles` for this tool,
so the loopback run recorded in the pull request used a CBCT belonging to
another tool. That proves the round trip, the checkpoint loading and the shape
of the output -- not that the canine is where it says.
