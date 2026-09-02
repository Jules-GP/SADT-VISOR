# sadt-ali

Places anatomical landmarks and writes Slicer markups files (`.mrk.json`). One
tool, two engines that share nothing but their output format:

- **CBCT** -- one deep-RL agent per landmark walks the volume at 1 mm and then at
  0.3 mm until it converges on the point. 119 landmarks across four regions.
- **IOS** -- per tooth, the mesh is rendered from a dozen viewpoints and a 2D
  UNet predicts masks that are projected back onto the surface. Three networks:
  **Occlusal**, **Cervical**, and **Mucogingival** -- the last one on the
  gingival margin rather than the crown, mandible only, off by default.

Which engine runs is decided from the data, never from an argument.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, paths `ALI_CBCT/`,
`ALI_CBCT_utils/`, `ALI_IOS/` and `ALI_IOS_utils/`, by way of
`slicer-remote-tool-server`'s `tools/ALI/` -- whose history this repository
carries, so `git log --follow` on `src/sadt_ali/cbct/engine.py` reaches back
through it to `a0ed474` (2026-07-31).

> **The upstream commit is not recorded.** The server-side port landed as
> `ADD ALI & CrownSeg` with no upstream SHA in the message and none anywhere in
> the tree, so there is no way to recover from this repository which upstream
> revision the algorithm was taken from. Every other row in
> [PROVENANCE.md](../../PROVENANCE.md) carries one. **This needs filling in by
> whoever made that port**, and until it is, "same as upstream" is a claim about
> code nobody can point at. The per-module mapping below is exact and was
> verified by reading; only the revision is missing.

| This package | Upstream |
|---|---|
| `cbct/engine.py` | `ALI_CBCT/ALI_CBCT.py` |
| `cbct/agent.py` | `ALI_CBCT_utils/agent.py` |
| `cbct/brain.py` | `ALI_CBCT_utils/brain.py` |
| `cbct/environment.py` | `ALI_CBCT_utils/environment.py` |
| `cbct/preprocess.py` | `ALI_CBCT_utils/preprocess.py` |
| `ios/engine.py` | `ALI_IOS/ALI_IOS.py` |
| `ios/surface.py` | `ALI_IOS_utils/surface.py` |
| `ios/render.py` | `ALI_IOS_utils/{render,mask_renderer,agent}.py` |

Upstream pins **not** kept: this package pins torch 2.8.0+cu128, monai 1.6.0,
itk 5.4.7 and Python 3.11. See "Versions" for why.

Changes from upstream -- the algorithm is untouched, the envelope is not. The
first four were made during the server-side port and are unchanged here; the
rest are this migration's.

- **One markups file per scan**, holding every landmark found. Upstream wrote
  one file per anatomical region, so every downstream tool (ASO, AREG,
  AutoMatrix) had to recombine them by hand.
- **`display.visibility` is `true`.** Both upstream CLIs wrote `false`, which
  switches the markups *display node* off: Slicer loads the file, lists the node
  and draws nothing. Invisible inside the old Slicer module, which loaded nodes
  itself; fatal for anyone opening a returned file.
- **Scans are keyed by path relative to the input root**, not by base name. Two
  patients called `scan.nii.gz` in different folders used to overwrite each
  other, twice -- in the working dictionary and again in the flat output folder.
- **Both impacted-canine spellings resolve** (`UR3OI` ≡ `UR3OIP`) and
  `group_of()` never raises. The unguarded `LABEL_GROUPS[...]` lookup this
  replaces threw a `KeyError` caught far above, and *nothing at all* was written
  for that scan -- including every landmark already found.
- **Zip extraction removed.** The server unpacks archives before `run()` is
  called, with the bomb cap and `strip_single_root` that used to live in
  `ALILogic._extracted`.
- **Model-bundle auto-selection removed.** `model` was optional and the tool
  picked a hosted bundle matching the detected mode by walking `data_store`. A
  tool no longer resolves paths, so `model` is required and the server picks.
  The layout check survives: a bundle of the wrong kind is still refused with a
  message naming both kinds, which is what that code was really for.
- **Per-tooth selection is not exposed.** Upstream's IOS CLI takes `teeth` and
  `teeth_mg` -- which teeth to predict on, and which to predict the mucogingival
  point for. `ALI_IOS` takes neither: every tooth the mesh carries a label for is
  predicted, on every run.

  Believed safe to omit for now, and stated rather than dropped in silence. The
  networks run per tooth either way, so the selection buys time rather than a
  different result, and a caller that wants a subset can filter the markups file
  it gets back. What it would cost is a batch where only a few teeth are of
  interest: that run does the full arch and pays for it.

  If that turns out to matter, this is a stated gap to reopen rather than a
  rediscovery. Found by comparing against upstream d5a48c4 (2026-08-18); see the
  same audit's note on the six inference-tuning parameters (`spacing`,
  `agent_FOV`, `spawn_radius`, `image_size`, `blur_radius`, `faces_per_pixel`),
  which are deliberately not exposed either.

- **CrownSeg is no longer called.** `ALILogic.ensure_segmented()` imported
  `tools.CrownSeg` in-process and segmented an unlabelled mesh on the fly.
  Tools do not call each other; `ios.engine.require_labels()` refuses the batch
  up front instead, naming `Crown_Seg` and the array it looked for. **This is
  the one behaviour change a user can see** -- see "The Crown_Seg chain".
- **The GPU semaphore is gone.** Both engines held a
  `threading.BoundedSemaphore(ALI_MAX_GPU_JOBS)`, which serialised inference
  when every tool shared one process. A tool is its own process now, so an
  in-process limit caps nothing; the server holds it, across tools.
- **`device` and `search_seconds` are arguments**, not settings. `run()` must
  not read the environment.
- **`ui="tabs"` and the `groups` table are gone**, with the `ArgSpec` schema
  that carried them. See "What the client loses".

## What it does

| | |
|---|---|
| Inputs | `input`: one CBCT (`.nii`/`.nii.gz`/`.nrrd`/`.nrrd.gz`/`.gipl`/`.gipl.gz`), one intraoral surface (`.vtk`/`.stl`), or a folder of either, searched recursively. A DICOM series inside a folder is converted automatically. `model`: the bundle. `output_dir`: where results go. |
| Outputs | One `<scan>_lm_<ID>.mrk.json` per scan, mirroring the input's folder tree, plus `run_report.json`. |
| Model files | CBCT: `<bundle>/**/<landmark>/<scale>/*.pth`, scale folders named `1` and `0-3`, both required per landmark. IOS: flat checkpoints carrying an `O`/`C` token and an `Upper`/`Lower` one, e.g. `Upper_O_model.pth`. Fetched by the server into `/DATA/ALI/models`. |
| GPU | Used when available; `device="cpu"` works and is much slower -- the per-landmark search budget defaults to 60 s on CPU against 15 s on CUDA for that reason. |

Three behaviours worth knowing before reading a result:

- **`landmarks` replaces `cbct_regions`, it does not narrow it.** Naming any
  landmark makes the region selection inert. That is what lets a caller ask for
  the seven points it needs instead of running 58 agents to use seven -- one
  agent being a full two-scale walk of the volume.
- **A landmark missing from the bundle and a landmark that never converged are
  different things**, and look identical in the Slicer scene. The first is in
  `landmarks_without_model`, the second in that scan's `landmarks_failed`. They
  need opposite fixes: another bundle, or another scan.
- **An input holding both CBCT and IOS data is refused**, not half processed.

## The Crown_Seg chain

ALI's IOS engine needs meshes that already carry tooth labels. Upstream, and in
the server-side port, ALI segmented them itself by importing CrownSeg. It does
not any more, so the sequence is the server's to run:

```
Crown_Seg  →  ALI (IOS)
```

Crown_Seg's `run_report.json` lists every labelled mesh under `segmented_meshes`,
whether this run produced the labels or found them already there -- so re-running
the chain on a mixed batch is cheap and safe. Feed its output directory straight
in as ALI's `input`.

Skipping it is not a silent failure: `require_labels()` checks the whole batch
before any weights load and refuses it with a message naming `Crown_Seg` and the
three array names it looked for. Checking up front matters -- discovering it on
mesh 40 of 40 costs an hour of inference first.

`tests/test_integration.py` runs the real chain, each tool in its own venv, the
way the server does.

## What the client loses

The old `ArgSpec` schema published presentation metadata that `describe.py` has
no field for. Two are worth stating plainly, because a client written against
the old schema will look worse rather than break:

- **`landmarks` had `ui="tabs"` and a `groups` table**, so 119 check boxes
  rendered as four tabs matching the anatomical regions. The new schema
  publishes the 119 options as a flat `choices` list. The grouping still exists
  in `cbct.catalog.GROUP_LABELS`; nothing publishes it.
- **`section` is gone**, so the four collapsible boxes ("Inputs", "CBCT
  landmarks", "IOS landmarks", "Outputs") are no longer declared. Both
  selections are still always shown, and one is always inert.

Neither affects a result. Both affect how usable the panel is for the one tool
in this repository with a three-figure option count, and both are the same
question: whether the schema should carry layout hints at all. Raised here
rather than worked around.

## Versions

Pinned to what the deployed server actually runs -- torch 2.8.0+cu128,
monai 1.6.0, itk 5.4.7, SimpleITK 2.5.6, vtk 9.6.2, numpy 2.3.2, Python 3.11 --
which is the same reasoning as [AMASSS](../AMASSS/README.md#versions): the
Slicer module installs into Slicer's shared interpreter, where the pins are a
truce with fifteen other modules, and no ALI result has ever been produced on
them. monai 1.6.0 and itk 5.4.7 are what the sibling tools already lock, so this
adds no runtime to the image.

The one place a version is load-bearing in the code:
`cbct/environment.py` uses monai's `EnsureChannelFirst`. Upstream branched on
`sys.version_info >= (3, 10)` to choose between it and `AddChannel`, which monai
removed years ago; at 1.6.0 only the former exists and the branch is gone.

### pytorch3d

`ALI_IOS` needs it and `ALI_CBCT` does not, which is the whole reason the two
were split. It is a **direct, pinned dependency of `ALI_IOS` alone**:

```toml
dependencies = ["torch==2.11.0", "torchvision==0.26.0", "pytorch3d==0.7.9+pt2110cu128", ...]

[[tool.uv.index]]
name = "pytorch3d-wheels"
url = "https://ImageMindAnalytics.github.io/pytorch3d-wheels/simple/"
explicit = true

[tool.uv.sources]
pytorch3d = { index = "pytorch3d-wheels" }
```

**A prebuilt wheel, not a source build.** That index is the one upstream's own
`install_pytorch.py` reads, and `uv.lock` pins the `cp311 manylinux_2_28`
wheel by sha256. A plain `uv sync --frozen` installs it: no extra, no `nvcc`,
no compilation, and the deployment image gets it the same way as any other
package.

The wheel tag `+pt2110cu128` names one torch version and one CUDA variant
exactly, which is why torch and torchvision are pinned beside it -- the three
move together or the C extension does not load. `explicit = true` on both
indexes is load-bearing: as a plain extra index uv is free to mix registries,
which is how a torch 2.11.0+cu130 once got paired with a wheel that imports and
then dies on a missing libcudart.

## Validated against

- **Against upstream, 2026-09-02, both engines.** `origin/main` extracted and
  run with this package's own virtualenv, on the same mesh, the same bundle and
  the same rasterisation settings (224 / 0 / 1).

  *IOS*: 73 landmarks each side, the same labels, **54 identical to the
  float**. All 19 that differ land on the correct tooth, and 17 of them come
  from one line: upstream truncates its logits to `int16` before the argmax on
  the crown networks, which turns near ties into exact ties that argmax
  resolves toward the background. Measured over the Cervical pass, 12 576
  pixels of 15.65 M change class -- the landmark channels losing 6 283 and
  gaining 145, so the mask only ever shrinks, and it shrinks more the higher
  the class index (`CB` is class 2 and loses ties to `CL` as well as to the
  background: it drops 31-74 % of its faces, `CL` 14-47 %). Restoring that one
  cast takes the port to **71 of 73 identical**; the two residuals are
  `LR2CL` at 0.198 mm, below this mesh's own vertex spacing, and `LR5MG` at
  0.179 mm from the per-tooth camera aiming.

  *CBCT*, on Ba/S/N: two identical to the float, `N` differing by 0.187 mm.
  Upstream's bounds check reads `new_pos.all() > 0`, which reduces the vector
  to one boolean before comparing -- so it tests "no component is exactly
  zero" rather than "every component is positive", and a legitimate coordinate
  of zero respawns the agent at a random position. Both defects date from the
  original implementation, July 2022, and are still on `origin/main`.

- **Schema**: `describe.py` accepts both signatures. `ALI_CBCT` publishes 8
  arguments, with `choices` on `regions` (4), `landmarks` (119) and `device`
  (2); `ALI_IOS` publishes 6, with `choices` on `networks` (3) and `device`
  (2). Asserted out of process against the real venvs.
- **Tests**: 50 in `ALI_CBCT` (2 GPU tests deselected) and 12 in `ALI_IOS`. The agent is stubbed,
  so mode detection, DICOM recognition, weight discovery, the vocabulary, output
  naming, tree preservation, the run report, work-directory cleanup, the
  output-containment rule and every cross-argument rule run for real, with no
  checkpoint and no card.
- **Weight discovery, against the real 14 GB bundle**: `discover_weights` finds
  **119 landmarks carrying both scales** -- exactly the 119 this package's
  catalog declares, with nothing left ungrouped, so the bundle's folder names
  and the vocabulary agree completely. The IOS bundle resolves all four
  (network, jaw) pairs and reports `Lower_MG_v6.pth` as unrecognised, which is
  the naming rule doing its job.
- **Against the pre-port implementation: bit-identical.**
  - **Input**: `DATA/ALI/testfiles/MG_test_scan.nii.gz`, one real CBCT.
  - **Weights**: `DATA/ALI/models/ALI_CBCT_Models`, the seven landmarks ASO
    registers on (`Ba`, `S`, `N`, `RPo`, `LPo`, `ROr`, `LOr`) -- the set that
    matters most, since another tool depends on it.
  - **Reference**: `slicer-remote-tool-server`'s `tools/ALI/`, verified
    byte-identical to the revision this repository imported, run **inside this
    package's own venv** with the four server modules stubbed. So torch
    2.8.0+cu128, monai 1.6.0, itk 5.4.7, the weights and the card are the same
    on both sides and the *only* variable is the code.
  - **Result**: each implementation run twice, all 16 port×reference pairs
    compared. **Every landmark matches to the full float, 0.0000 mm.** The
    reference's own run-to-run spread is also 0.0000 mm, so this pipeline is
    deterministic and "identical" means identical, not "within the noise".

    | | Max | Mean | Exact |
    |---|---|---|---|
    | reference vs reference | 0.0000 mm | 0.0000 mm | 7/7 |
    | port vs port | 0.0000 mm | 0.0000 mm | 7/7 |
    | port vs reference (×4) | 0.0000 mm | 0.0000 mm | 7/7 |

  - **Tolerance**: none needed. **Treat any non-zero difference as a
    regression** -- unlike AMASSS, where nnUNet's CUDA nondeterminism sets a
    noise floor, nothing here is nondeterministic on a scan whose agents
    converge without leaving the volume.
- **GPU tests were run**: `uv run pytest -m "gpu and models"` on an RTX 6000 Ada
  -- passed in 38 s, all seven landmarks found, `device=cuda`, every point inside
  the scan's own physical extent.
- **The IOS half, on real weights and a real card** -- run inside
  `ghcr.io/jules-gp/lab-ai:2026.08`, which already carries pytorch3d 0.7.9 (the
  tag this package pins) and CUDA 12.8, so no source build was needed:
  - **Input**: `DATA/ALI/testfiles/T1_01_U_segmented.vtk`, a real segmented
    upper arch. **Weights**: `DATA/ALI/models/ALI_IOS_Models`.
  - **Occlusal**: **42 landmarks** -- 14 teeth × 3 types, every tooth the mesh
    carries -- in 21 s, `device=cuda`, one markups file, no failures.
  - **Mucogingival on that same maxilla**: correctly produces **nothing** and
    does not fail the run. `NETWORK_JAWS` restricts it to the mandible, so an
    upper arch is not a missing model but a question the network cannot be
    asked -- `jaws_without_model` stays empty and the occlusal pass is
    unaffected.
- **Mucogingival runs.** On `DATA/FlexReg/testfiles/Arches/Lower_arch.vtk`, a
  labelled lower arch carrying all 13 MG teeth, it places 13 landmarks out of
  13 in 10 s on cuda, none of them forced. Twelve are identical to what
  upstream places; the thirteenth differs by 0.179 mm, from this package's
  per-tooth camera aiming.
- **What is NOT established is accuracy.** No annotated intraoral scan is
  staged here, so nothing measures how far these points sit from where a
  clinician would put them. The comparison below says the port is faithful to
  the code it came from -- not that the answer is right.

## Working on it

```bash
cd tools/ALI/ALI_IOS        # or ALI_CBCT -- tools/ALI/ has no pyproject of its own
uv sync                     # pytorch3d included, as a wheel
uv run pytest -m "not gpu"  # 12 tests here, 50 in ALI_CBCT
uv run pytest -m models     # needs the real bundles under DATA/ALI/models/
```

```bash
# The schema the server publishes, for this engine
.venv/bin/python ../../../scripts/describe.py .
```
