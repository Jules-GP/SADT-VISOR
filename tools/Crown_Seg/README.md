# sadt-crownseg

Labels every tooth of an intraoral surface scan with its dental number, as a
point-data array on a copy of the mesh. That array is the precondition for ALI's
IOS landmark identification and for the IOS modes of ASO, AREG and FlexReg,
which is why this is its own tool.

## Provenance

Not ported: written against `shapeaxi` directly. The Slicer modules ran the
`dentalmodelseg` executable out of Slicer's own bin directory, and that
executable is only the console-script entry point of the `shapeaxi` PyPI package
(`dentalmodelseg = shapeaxi.dental_model_seg:cml`). There is no Slicer binary to
shell out to here, so `shapeaxi.dental_model_seg.main` is called directly with
the namespace its own `cml()` would have built.

Carried over from `slicer-remote-tool-server`'s `tools/CrownSeg/`, whose history
this repository holds -- `git log --follow` on `src/sadt_crownseg/pipeline.py`
reaches back through it.

Changes made by this port:

- **ALI no longer calls this tool.** The server-side version existed partly so
  `ALILogic.ensure_segmented()` could import it and segment raw meshes inline.
  Tools do not call each other any more, so the server sequences CrownSeg before
  ALI, and the report's `segmented_meshes` list is what it reads to do it.
- **Scratch space lives under `output_dir`** (`.crownseg_work/`, removed in a
  `finally` so it goes whichever way the run ends -- it holds a csv listing the
  patient's own file paths), and `segment_crowns()` returns the run report
  rather than a `CrownSegRun`. Arguments are validated *before* anything is
  created, so a refused request leaves `output_dir` exactly as it found it.
- **Zip extraction removed** -- the server unpacks archives before `run()`.
- **The model is a required argument.** It used to be a name resolved through
  the data store, with `default_model_path()` reading `settings.CROWNSEG_MODEL`;
  weights now arrive as a `Path`, like every other tool.
- **The GPU semaphore is gone** -- each call is its own process.

## Working around a shapeaxi bug

**CrownSeg does not currently run in the deployment, and did not before this
port either.** `shapeaxi.dental_model_seg` -- the only supported way in -- reads
`DentalModelSeg` off `shapeaxi.saxi_nets`, but in the 2.0.x line the class lives
in `shapeaxi.saxi_nets_lightning`. Every 2.0.x release (2.0.0, 2.0.1, 2.0.2) has
this; 1.x referenced the right module. Running the pre-port tool in the deployed
container fails before it touches a mesh:

```
AttributeError: module 'shapeaxi.saxi_nets' has no attribute 'DentalModelSeg'
```

`_restore_moved_class()` in `pipeline.py` points the name back at where the class
went, before calling shapeaxi's `main()`. Two lines rather than reimplementing
`main()`, so the entry point stays the one shapeaxi supports; and it is guarded
by `hasattr`, so it becomes a no-op the day a release puts the name back. **This
should be reported at DCBIA-OrthoLab/ShapeAXI, and the function deleted once a
release carries the fix.** Two tests cover it: that it restores the name, and
that it does nothing when the name is already there.

## What it does

| | |
|---|---|
| Inputs | `meshes`: one `.vtk`/`.stl` surface or a folder of them, searched recursively. `model`: the checkpoint. `output_dir`: where results go. |
| Outputs | One labelled `.vtk` per mesh, named `<stem>_<suffix>.vtk`, plus `run_report.json`. The input tree is mirrored under `output_dir`; the meshes the network actually ran on are filed one level down, under `crownseg_input_<suffix>/`, which is shapeaxi's own csv-branch layout. `run_report.json` names every output, so a caller never has to know which branch a mesh took. |
| Model files | One checkpoint, `07-21-22_val-loss0.169.pth` (6.6 MB), from the Fly-by-CNN 3.0 release. |
| GPU | Used when available; `device="cpu"` works and is much slower. |

Three things worth knowing before reading a result:

- **An already-labelled mesh is passed through, not re-predicted.** That is what
  makes a mixed batch of raw and pre-segmented meshes one call. Pass
  `skip_segmented=False` to force the network to run anyway.
- **A pre-segmented `.stl` is impossible** -- STL carries no point data by
  construction -- so every output is `.vtk`, whichever branch a mesh took.
- **`numbering` changes the integers written into the array**, not the mesh.
  Whatever consumes the result has to agree; the report records which was used.

## Versions

torch 2.11.0+cu128, torchvision 0.26.0+cu128, pytorch3d 0.7.9+pt2110cu128,
vtk 9.6.2, numpy 2.3.2, Python 3.11.

These follow **upstream SlicerAutomatedDentalTools**, which is the reference:
its shared `shapeaxi` conda env is created with `ocnn==2.2.1`,
`shapeaxi>=2.0.2` and `SimpleITK`, and every pytorch3d-dependent module (ALI,
AREG, ASO, DOCShapeAXI, FlexReg) ships an `install_pytorch.py` that reads
<https://ImageMindAnalytics.github.io/pytorch3d-wheels/simple/> and picks the
wheel whose `+ptXXXcuYYY` tag matches the installed torch. The earlier
`torch==2.8.0` here came from this port, not from upstream.

```toml
[project.optional-dependencies]
segmentation = [
    "shapeaxi>=2.0.2", "ocnn==2.2.1", "SimpleITK",
    "pytorch3d==0.7.9+pt2110cu128", "torchvision==0.26.0",
]
```

**We are deliberately stricter than upstream on one point**: upstream leaves
torch unpinned, so it arrives as a shapeaxi dependency and resolves to whatever
is current. We pin `torch==2.11.0`. An unpinned torch is not reproducible -- the
same `uv sync` two months apart gives two different runtimes -- and the pytorch3d
wheel tag names one torch version and one CUDA variant *exactly*. Bumping torch
therefore means bumping the pytorch3d tag in the same commit; they are one
decision, not two.

### Python 3.11, not 3.12

Upstream moved this env to 3.12 because shapeaxi 1.0.10 pinned
`grpcio==1.51.1`, which has no cp312 wheel. shapeaxi 2.0.2 dropped that pin, so
grpcio resolves to 1.83.0, which publishes cp311 wheels -- the reason for the
move no longer applies. Verified rather than assumed: the full set
(shapeaxi 2.0.2, ocnn 2.2.1, torch 2.11.0, pytorch3d, monai, itk) resolves on
3.11 in under a second.

### The four numbers that must agree

pytorch3d and torchvision both ship compiled extensions linked against one
specific torch build. Three numbers have to line up -- torch version, CUDA
variant, Python ABI -- and a mismatch is **not** an install error: the package
imports fine and dies on the first CUDA kernel.

```toml
[[tool.uv.index]]
name = "pytorch3d-wheels"
url = "https://ImageMindAnalytics.github.io/pytorch3d-wheels/simple/"
explicit = true      # a PINNED SOURCE, never --extra-index-url

[tool.uv.sources]
torch       = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
pytorch3d   = { index = "pytorch3d-wheels" }
```

- `explicit = true` is what makes these pinned sources. As extra index URLs uv
  is free to mix registries: a torch `2.11.0+cu130` paired with a
  `pt2120cu126` wheel, failing at runtime on a missing `libcudart.so.12`.
- **`torchvision` is listed as a direct dependency**, for the same reason
  `pytorch3d` is: `tool.uv.sources` applies to direct dependencies only. Left
  transitive through shapeaxi it comes from PyPI, and PyPI's torchvision is
  built against the default torch rather than the cu128 one. The version
  pairing is right (0.26 ↔ 2.11) and it still fails, with
  `RuntimeError: operator torchvision::nms does not exist`. Measured here, not
  anticipated.
- pytorch3d is no longer compiled from git. Upstream installs a prebuilt wheel
  and so do we, which removes the CUDA toolkit and the tens of minutes.

`uv lock` resolves 172 packages in under a second.

### Verified on the GPU

An import proves nothing; the wheel has to run a kernel:

```
torch 2.11.0+cu128   CUDA 12.8   NVIDIA RTX 6000 Ada Generation
torchvision 0.26.0+cu128         torchvision::nms registered
pytorch3d 0.7.9+pt2110cu128      knn_points on cuda -> OK
```

End to end on `T1_01_U_segmented.vtk` (294,260 points) with
`skip_segmented=False` to force the engine: **52.3 s**, 15 distinct tooth
labels, and `PredictedID` **bit-identical to the reference -- 0 of 294,260
points differ**.

`PredictedID` is the array to compare, and the only one. It is the network's
raw per-point prediction; `Universal_ID` is shapeaxi's post-processed
labelling, which the section below shows is not deterministic between two runs
of the *same* code. Bit-identical `PredictedID` under torch 2.11 + pytorch3d
0.7.9+pt2110cu128 therefore says the version move does not perturb inference at
all -- a stronger statement than any agreement percentage, and the same result
the pre-port validation recorded against torch 2.8.

`uv sync --extra segmentation` no longer compiles anything: the wheel is
prebuilt, so there is nothing to cache in a local `/wheels` directory.

## Validated against

- **Input**: `DATA/CrownSeg/testfiles/T1_01_U_segmented.vtk` (294 260 points),
  with `skip_segmented=False` so the network actually runs -- the published test
  mesh is already labelled.
- **Weights**: `07-21-22_val-loss0.169.pth`, staged from the release named in the
  server's manifest and checked against its recorded size.
- **Reference**: the pre-port implementation **plus the same shapeaxi
  workaround**, since without it the reference cannot run at all. Both were run
  inside the deployed container, so the engine, the weights and the GPU are
  identical and only the tool code differs.
- **Result**: same geometry (294 260 points, 588 244 cells, points bit-identical)
  and the same two output arrays.

| Array | What it is | Port vs reference |
|---|---|---|
| `PredictedID` | the network's raw per-point prediction | **bit-identical**, 0 of 294 260, in all four runs |
| `Universal_ID` | shapeaxi's post-processed labelling | varies run to run, including between two references |

```
Universal_ID: differing points of 294260
            ref1    ref2    ref3    port
ref1           0    2174    1216     930
ref2        2174       0    2361    2607
ref3        1216    2361       0     826
port         930    2607     826       0
```

- **Tolerance**: none of the difference is attributable to the repackaging.
  `PredictedID` -- everything the network produces -- is identical every time, so
  the port demonstrably does not touch inference. `Universal_ID` comes out of
  shapeaxi's own "closing operation", which is not deterministic: two
  *reference* runs differ by 1 216-2 361 points, and the port's spread against
  them (826-2 607) is the same range. **Treat a difference in `PredictedID` as a
  regression**; `Universal_ID` alone is this noise floor.

**GPU tests**: four tests are marked `models`, two of them also `gpu`, and both
markers are **deselected by default** (`addopts = "-m 'not gpu and not models'"`)
rather than skipped -- a green line that is half skips says nothing. They now run
through pytest rather than by hand: pytorch3d arrives as a prebuilt wheel, so
there is no CUDA toolkit to want. Last run, on the real checkpoint and
`T1_01_U_segmented.vtk`:

```
SADT_CROWNSEG_MODEL=.../07-21-22_val-loss0.169.pth \
SADT_CROWNSEG_MESH=.../T1_01_U_segmented.vtk \
uv run pytest -m models -o addopts=      # 4 passed in 119s
```

The other 176 tests stub the engine and run anywhere, which is what CI does.

## Working on it

```bash
cd tools/Crown_Seg
uv sync                          # fast: no shapeaxi, no pytorch3d
uv run pytest                    # 176 tests, engine stubbed
uv sync --extra segmentation     # prebuilt pytorch3d wheel, nothing compiles
uv run pytest -m models -o addopts=   # see tests/data/README.md
```

```bash
# The schema the server publishes
.venv/bin/python ../../scripts/describe.py .
```
