# sadt-batchdentalseg

Segments teeth and jaw structures on a dental CT or CBCT, with the
DentalSegmentator family of nnUNet v2 models. One bundle per model; the bundle
you pick chooses the label table with it.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, path
`BATCHDENTALSEG/BATCHDENTALSEGLib/SegmentationWidget.py`, commit `6df3fab`
(2026-08-05), by way of `slicer-remote-tool-server`'s `tools/BatchDentalSeg/` --
whose history this repository carries, so `git log --follow` on
`src/sadt_batchdentalseg/pipeline.py` reaches back through it.

Upstream pins **not** kept, for the same reason as AMASSS: this package pins
torch 2.8.0+cu128 and nnunetv2 2.8.1, the stack the deployed server runs. See
"Versions".

Upstream is a 2940-line Qt widget and most of it is not this pipeline. Already
absent before this port, each for a stated reason: the queue table, the RAM
watchdog, killing nnUNet processes a crashed scan left behind, the "free
memory" button, the per-scan cool-down, restoring the queue from disk -- all of
which exist because the widget runs inside Slicer on a clinician's laptop and
has to survive being out of memory. Also not ported: the runtime model download
from GitHub releases (a tool holding patient data does not make outbound calls
mid-run), the auto-crop (upstream applies it only when its RAM preflight fails,
and it changes what the network sees), the mirroring resolution (a button the
user presses after looking at the result), and the mesh exports.

Changes made by this port -- the algorithm is untouched:

- **Scratch space lives under `output_dir`** (`.batchdentalseg_work/`, removed
  before returning). A tool must not write outside the directory it is given.
- **`segment()` returns the run report** instead of a `SegmentationRun`; tools
  no longer call each other.
- **Zip extraction removed.** The server unpacks archives before `run()`.
- **The GPU semaphore is gone** -- each call is its own process now.
- **`device` is a `Literal["cuda", "cpu"]`**, so the schema publishes both
  options. `model` deliberately stays a plain `Path`: which bundles exist is a
  property of the deployment, not of this package, so the picker comes from the
  server's data listing rather than from a hard-coded set here that would go
  stale the moment a bundle is not staged.
- **`device` and `tile_step_size` are arguments**, not server settings.
  `run()` must not read the environment, and `tile_step_size` moves the
  segmentation. `BATCHDENTALSEG_MAX_GPU_JOBS` went with the semaphore.
- **`check_dependencies()` is gone.** It existed so a deployment missing torch
  reported that once per run rather than once per scan; the lockfile makes it
  unreachable.

## What it does

| | |
|---|---|
| Inputs | `scans`: one scan (`.nii`/`.nii.gz`/`.nrrd`/`.nrrd.gz`/`.gipl`/`.gipl.gz`) or a folder of them, searched recursively. `model`: the bundle. `output_dir`: where results go. |
| Outputs | One `<scan>_<ID>` label volume per scan, mirroring the input tree, plus `BatchDentalSeg_report.json`. With `separate_segments`, one binary file per label present. |
| Model files | A bundle directory holding `dataset.json`, `plans.json` and `fold_0/checkpoint_final.pth`. Fetched by the server into `/DATA/BatchDentalSeg/models`; see `tests/data/README.md` for staging one by hand. |
| GPU | Used when available; `device="cpu"` works and is much slower. |

Models, and what each labels:

| Bundle | Segments |
|---|---|
| `DentalSegmentator` | Adult. Upper Skull (maxilla included), Mandible, Upper Teeth, Lower Teeth, Mandibular canal |
| `PediatricDentalSeg` | Paediatric, the same five |
| `NasoMaxillaDentSeg` | Six -- the maxilla is split out of the Upper Skull, which shifts every later value |
| `UniversalLab` | Every tooth individually in Universal numbering, deciduous included, plus Mandible, Maxilla and Mandibular canal |

Three things worth knowing before reading a result:

- **The bundle's folder name selects the model**, and the label table follows
  from it. That is deliberate: a separate "which labels" argument would let a
  caller pair bundle X with the labels of Y, and the result would be a
  plausible volume with every structure named wrong.
- **The label values are part of the trained weights**, not a presentation
  choice -- they are the integers the network emits. Renaming a catalog entry is
  safe; renumbering one silently mislabels anatomy. The report ships the table
  next to the results, because the segmentation is a volume of integers and
  without it they mean nothing.
- **`separate_segments` writes only labels PRESENT in the scan.** A full
  UniversalLab run would otherwise produce 55 files per patient, most empty, and
  an empty mask is indistinguishable from a structure the model failed on.

### A cross-repo contract this repository cannot enforce

A key in `catalogs.MODELS` **must equal** the folder name the server's
`scripts/data-manifest.yml` downloads that bundle into. A key that drifts makes
an installed model unselectable. The server-side suite asserted this against the
manifest directly; that file lives in the other repository and a tool package
cannot reach it, so the check is gone and only this note remains.

## Versions

torch 2.8.0+cu128, nnunetv2 2.8.1, SimpleITK 2.5.6, numpy 2.3.2, Python 3.11 --
the stack the deployed server runs, which is where every BatchDentalSeg result
to date was produced. No vtk: the mesh exports are not ported.

The CUDA wheels come from an explicit index (`explicit = true` is load-bearing --
without it uv looks for every package on the PyTorch index); the venv is 7.2 GB.
The reasoning is the same as AMASSS's, at length in `tools/AMASSS/README.md`.

## Validated against

- **Input**: `DATA/AMASSS/testfiles/MG_test_scan.nii.gz`, one real CBCT.
- **Weights**: `PediatricDentalSeg`, staged from the upstream release named in
  the server's manifest and checked against its recorded sha256s and sizes.
- **Reference**: the pre-port implementation, run inside the deployed server
  container on the same scan, bundle and settings, on the same GPU, with
  `separate_segments=True`.
- **Result**: same output tree, same file names, identical `labels` table and
  identical `model` / `device` / `tile_step_size` / `separate_segments` /
  `summary` report fields.

As with AMASSS, nnUNet on CUDA is not bit-deterministic, so the reference was
run three times and this package three times before any conclusion was drawn.
Both sets scatter across the same states -- 2 of the 9 port×reference pairs are
bit-identical on the label volume -- and the worst port-vs-reference difference
is the *same number* as the worst reference-vs-reference difference on five of
the six files:

| File | Voxels | Worst ref vs ref | Worst port vs ref | Worst Dice |
|---|---|---|---|---|
| `_Seg` (labels) | 4 829 309 | 180 | 180 | 0.999982 |
| `_Upper-Skull` | 3 096 518 | 144 | 144 | 0.999977 |
| `_Mandible` | 1 223 918 | 34 | 35 | 0.999986 |
| `_Upper-Teeth` | 262 881 | 6 | 6 | 0.999989 |
| `_Lower-Teeth` | 227 033 | 6 | 6 | 0.999987 |
| `_Mandibular-canal` | 18 959 | 1 | 1 | 0.999974 |

- **Tolerance**: **Dice < 0.9999 against a reference is a regression**;
  above it is nnUNet's own CUDA noise floor, which the pre-port tool shares.

**GPU tests were run**: `uv run pytest -m models` on an RTX 6000 Ada -- passed,
with the PediatricDentalSeg bundle when this was ported and again with
DentalSegmentator when `gpu_resampling` was added (that one runs the scan both
ways and compares every label). CI skips them (`-m "not gpu"`); the other 38
tests stub `nnunet_runner.predict_folder` and need no checkpoint.

## GPU resampling

`gpu_resampling` (default **on**) points nnUNet's two resamplers at the card
instead of its scipy splines. AMASSS took this change first and this package
deliberately did not, on the grounds that nothing had measured what it costs
THESE models. This is that measurement.

**Setup.** `DATA/AMASSS/testfiles/MG_test_scan.nii.gz`, one real CBCT,
512x512x365 at 0.33 mm -- the same scan AMASSS was profiled on, so the two sets
of numbers are comparable. RTX 6000 Ada, 48 GiB. All four bundles, one scan
each, `tile_step_size` at its default. Every bundle's plans name nnUNet's stock
`resample_data_or_seg_to_shape` at both ends, so the swap applies to all four.

### Where the time goes

Seconds, one scan. "in" is the input volume resampled to the model's grid,
"out" the logits resampled back, "crop" the nonzero mask nnUNet crops by --
which is NOT swapped and stays on scipy in both columns.

| Bundle | classes | before | after | | in | crop | net | out |
|---|---|---|---|---|---|---|---|---|
| `DentalSegmentator` | 5 | **58.2** | **19.1** (3.05x) | scipy | 13.8 | 4.5 | 3.8 | 21.0 |
| | | | | torch | 0.2 | 4.5 | 3.8 | 1.1 |
| `PediatricDentalSeg` | 5 | **57.5** | **19.5** (2.95x) | scipy | 13.3 | 4.4 | 4.1 | 20.2 |
| | | | | torch | 0.2 | 4.5 | 4.0 | 1.1 |
| `NasoMaxillaDentSeg` | 6 | **70.9** | **46.4** (1.53x) | scipy | 14.0 | 4.5 | 9.5 | 24.2 |
| | | | | torch | 0.4 | 9.7 | 11.1 | 2.5 |
| `UniversalLab` | 55 | **278.7** | **58.4** (4.77x) | scipy | 14.2 | 4.4 | 9.0 | 189.5 |
| | | | | torch | 0.4 | 9.0 | 10.0 | 10.9 |

The before/after columns are wall clock through `segment()` on the paths the
tool actually takes -- `predict_from_files` with its worker processes for
scipy, `predict_from_files_sequential` for torch. The per-phase columns come
from separate sequential runs, because the worker path does its resampling in
processes this one cannot time.

Three things that table says:

- **The network was never the cost.** Resampling outweighs it 4x on Naso, 8-9x
  on the two five-segment bundles and **23x on UniversalLab**, where one
  resampling of 55 probability channels takes 190 seconds on one core.
- **UniversalLab is the bundle that needed this most**, and it is the one the
  original decision deferred. 4.6 minutes a scan becomes 58 seconds.
- **Naso gains least (1.53x)** because the residual scipy call -- the crop mask,
  which is not swapped -- roughly doubles on the torch path, from 4.5 s to
  9.7 s, and its network is genuinely slower (a 192x256x256 patch). Swapping
  `resampling_fn_seg` as well is the obvious next thing to measure; it is not
  done here because AMASSS did not, and matching AMASSS exactly is what let
  this be a measurement of one change.

### What it costs, per label

Dice against the scipy pipeline on the same scan, **plus a repeated scipy run**
so the comparison is read against nnUNet's own CUDA noise rather than against
zero. Volumes are the label volume differences in mm3, at 0.0359 mm3 a voxel.

| Bundle | scipy vs scipy (floor) | scipy vs torch, worst label | mean |
|---|---|---|---|
| `DentalSegmentator` | **bit-identical** | 0.9912 Mandibular canal (+8 mm3) | 0.9945 |
| `PediatricDentalSeg` | 0.99998 | 0.9928 Mandibular canal (+6 mm3) | 0.9958 |
| `NasoMaxillaDentSeg` | 0.99994 | 0.9914 Upper Skull (+602 mm3) | 0.9944 |
| `UniversalLab` | 0.99988 | 0.9934 Mandibular canal (+5 mm3) | 0.9970 |

The floor being ~0.9999 -- and bit-identical for one bundle -- is what makes
the attribution clean: the 0.991 is the resampler, not the card.

`UniversalLab` labels 55 structures and 31 were present in this scan. Its worst
ten, all of them thin midline structures:

| Dice | Label | dV (mm3) | max surface deviation |
|---|---|---|---|
| 0.9934 | Mandibular canal | +5.0 | 0.33 mm |
| 0.9945 | Lower-left central incisor | -2.4 | 0.33 mm |
| 0.9947 | Maxilla | +734.0 | 3.13 mm |
| 0.9948 | Lower-right lateral incisor | +0.8 | 0.33 mm |
| 0.9953 | Lower-right central incisor | +0.9 | 0.57 mm |
| 0.9953 | Upper-left central incisor | +3.6 | 0.33 mm |
| 0.9966 | Upper-right central incisor | +1.1 | 0.33 mm |
| 0.9968 | Lower-left lateral incisor | +0.7 | 0.33 mm |
| 0.9968 | Mandible | +185.1 | 0.66 mm |
| 0.9972 | Upper-left lateral incisor | +0.4 | 0.33 mm |

Most maximum deviations are 0.33 mm, which is one voxel of this scan -- the
boundary moved by a voxel in places, which is what dropping the input
interpolation from cubic to linear does. The larger ones (Maxilla 3.13 mm,
Naso's Upper Skull 7.79 mm) are on structures whose boundary runs along the
edge of the field of view, the same place AMASSS's cervical vertebra was the
outlier.

**Against AMASSS's numbers, which is the comparison that licensed this:** its
worst was the cervical vertebra at 0.978 and its cranial base at 0.991. No
bundle here is worse than 0.991, and three of the four are better than AMASSS's
second-worst.

### Why it is on by default anyway, and when to turn it off

On, for all four bundles: every one gains (1.5x to 4.8x), no label of any
bundle falls below what AMASSS already accepted, and the largest win is on the
bundle that is otherwise unusable interactively.

**Memory is the one reason to turn it off, and it is UniversalLab's alone.**
Peak VRAM, one scan:

| Bundle | scipy | torch |
|---|---|---|
| `DentalSegmentator` | 2.8 GiB | 4.1 GiB |
| `PediatricDentalSeg` | 2.6 GiB | 4.1 GiB |
| `NasoMaxillaDentSeg` | 10.9 GiB | 10.9 GiB |
| `UniversalLab` | 15.5 GiB | **37.1 GiB** |

Resampling 55 probability channels on the card is what does that. It fits on a
48 GiB card with room for nothing else, so:

- on a card smaller than ~40 GiB, pass `gpu_resampling=false` for
  **UniversalLab**; the other three want ~4 GiB and are unaffected;
- two concurrent UniversalLab runs will not fit on one 48 GiB card either way,
  which is the server's `MAX_CONCURRENT_GPU_JOBS`, not this argument.

**The report records it.** `BatchDentalSeg_report.json` carries
`"gpu_resampling": true|false` beside `tile_step_size`, because a result
produced this way is not bit-identical to nnUNet's own pipeline and whoever
opens a segmentation has to be able to tell which one made it. It reads false
on a CPU run whatever was asked for.

## Working on it

```bash
cd tools/Batch_Dental_Seg
uv sync                    # ~7.2 GB, CUDA wheels
uv run pytest -m "not gpu" # 38 tests, no GPU and no checkpoints needed
uv run pytest -m models    # a real bundle, see tests/data/README.md
```

```bash
# The schema the server publishes
.venv/bin/python ../../scripts/describe.py .
```
