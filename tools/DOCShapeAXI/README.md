# DOCShapeAXI

Classify a 3D anatomical shape, and paint the reason for the classification
onto the surface.

A shapeaxi network renders each mesh from several viewpoints, predicts a class
(or a value, for the regression checkpoint), and captum's `LayerGradCam`
projects the attribution back onto the mesh as a point array — so the result is
not only a grade but a surface a clinician can look at.

Ported from `DOCShapeAXI` / `DOCShapeAXI_CLI` in SlicerAutomatedDentalTools.

## Arguments

| argument | meaning |
|---|---|
| `surfaces` | a `.vtk` surface, or a folder of them, searched recursively |
| `model` | the checkpoint. Its name is the whole analysis — see below |
| `output_dir` | filled in by the server with the job's own output folder |
| `explain` | also compute the GradCAM attribution (default on) |
| `output_suffix` | appended to each written surface's stem, default `_pred` |
| `device` | `cuda` or `cpu`, falling back to `cpu` with a warning |

## One argument instead of five

Upstream's CLI takes `data_type`, `task`, `model`, `nn` and `num_classes`
separately, and only five of their combinations exist:

| checkpoint | anatomy | task | classes | network |
|---|---|---|---|---|
| `airways_2_class.ckpt` | nasopharynx airway obstruction | binary | 2 | classification |
| `airways_4_class.ckpt` | nasopharynx airway obstruction | severity | 4 | classification |
| `airways_4_regress.ckpt` | nasopharynx airway obstruction | regression | 1 | **regression** |
| `condyles_4_class.ckpt` | mandibular condyle | severity | 4 | classification |
| `cleft_4_class.ckpt` | alveolar bone defect in cleft | severity | 4 | classification |

Pairing them by hand is how a caller gets a plausible wrong answer: ask for
`airways_2_class` with `num_classes=4` and the run completes, reporting a
severity grade the network never learned. `num_classes` also drives how many
GradCAM maps are produced, so a wrong value silently changes the explanation
too. One argument names the checkpoint and `catalog.py` reads the rest.

The same reasoning is why `Batch_Dental_Seg` has one `model` argument rather
than a bundle plus a label table.

## What changed against upstream

- **A regression checkpoint no longer reports class 0 for every subject.**
  Upstream ran `argmax` on every network's output; a regression head has one
  output column, whose argmax is 0 for everyone. Every patient was graded
  identically, with no error anywhere.
- **Each surface is written once, under its own name.** Upstream wrote inside
  the per-class loop to a path that did not depend on the class, so a
  four-class run wrote the same file four times.
- **The file list is built in memory.** Upstream wrote a CSV into the *output*
  folder and read it back, opened with mode `'a'` — so a second run into the
  same folder appended to the first run's list.
- **`analysis_for()` raises rather than falling through.** Upstream's
  `find_model_name` had no final `else`: an unrecognised checkpoint logged "no
  model found for undefined task" and continued to a `NameError`. The message
  now names all five checkpoints that do exist.
- **The catalog is keyed by the file name that is actually published.**
  Upstream's `find_model_name` returns `clefts_4_class`, with an s, while the
  release asset is `cleft_4_class.ckpt`. The spelling upstream uses resolves to
  nothing.
- **A flat attribution map normalises to zeros, not to NaN.** Upstream divided
  by `max - min` unconditionally and wrote the NaNs onto the surface, where
  Slicer colours them as if they meant something.
- **The network is resolved from `shapeaxi.saxi_nets_lightning`.** This is why
  the prediction loop here is upstream's rather than shapeaxi's own
  `saxi_predict`: `saxi_predict` does `getattr(saxi_nets, args.nn)`, while both
  `SaxiMHAFBClassification` and `SaxiMHAFBRegression` live in
  `saxi_nets_lightning`. It raises `AttributeError` before touching a mesh —
  the same defect shapeaxi fixed for `dentalmodelseg` in 2.0.3 and has not
  fixed here.

## Not ported, each for a reason

- **`gradcam_save`** — dead upstream: it calls `shutil` without importing it,
  so it raises `NameError` on its first line that matters.
- **`MultiHead` and `SelfAttention`** — defined and referenced by nothing.
- **`download_model`** — fetches the checkpoints from a GitHub release
  mid-request. A server holding patient data does not make outbound calls
  during a request; the five checkpoints (343 MB) are staged under
  `DATA/DOCShapeAXI/models/` and selected by name.

## shapeaxi is pinned to 1.1.1, and upstream cannot load its own weights

`MHAEncoder.__init__` changed signature at shapeaxi **2.0.0**:

| | signature |
|---|---|
| 1.x | `input_dim, embed_dim, hidden_dim, num_heads, K, output_dim, sample_levels, dropout` |
| 2.x | `input_dim, output_dim, stages, num_heads, dropout, pooling_factor, pooling_hidden_dim, …` |

All five published checkpoints carry the **1.x** hyperparameters — every one of
them has `embed_dim=256`, `hidden_dim=64`, `num_heads=256`, `K=128` — so 2.x
answers `TypeError: MHAEncoder.__init__() got an unexpected keyword argument
'hidden_dim'` on all five.

Upstream's own installer asks for `shapeaxi>=2.0.2`
(`DOCShapeAXI_utils/install_pytorch.py:186`), which resolves to 2.0.3. **As
shipped, upstream DOCShapeAXI cannot load a single one of its own released
checkpoints.** Pinned here to `shapeaxi==1.1.1`, verified by loading all five.

That pin is also where numpy comes from: shapeaxi 1.1.1 requires `numpy<2.0`,
which is why upstream's unpinned conda environment resolves `captum==0.8.0`
(numpy<2 as well). The two agree. A tool having its own virtualenv is what
lets this one sit on numpy 1.26 while `Crown_Seg` next door runs shapeaxi 2.x
on numpy 2.

## The EfficientNet backbone is staged, not downloaded

shapeaxi builds a 2D EfficientNet-B0 around the mesh renderer, and it is *not*
in the checkpoint: constructing a network calls `EfficientNet.from_pretrained`,
which fetches `efficientnet-b0-355c32eb.pth` (20.4 MB) from GitHub. That is an
outbound call from inside a request, on a server holding patient data.

`check_backbone_is_staged()` refuses instead, naming the file, the URL and the
exact cache path. Point `TORCH_HOME` at a staged directory, or place the file
under `<torch hub dir>/checkpoints/`.

## The checkpoints are unpickled against an allowlist

torch 2.6 changed `torch.load`'s `weights_only` default to `True`, and these
checkpoints pickle their whole training transform pipeline into
`hyper_parameters`. `weights_only=False` would trust everything the file
contains; `allow_checkpoint_globals()` allowlists exactly the
`shapeaxi.saxi_transforms` classes plus `torchvision`'s `Compose`, and nothing
else.

## OpenCV is not a dependency

Upstream imports the whole of OpenCV for one call: `cv2.resize` of a 2D float
attribution map up to the renderer's 224×224 frame.

`torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)`
uses the same half-pixel convention as `INTER_LINEAR`, so it is the same
arithmetic from a library that is already here. Measured against a real
OpenCV over 36 cases of assorted input sizes: **worst relative difference
2.2e-05**, and **1.5e-05 absolute** through the whole `scale_attribution`
pipeline whose output range is [-1, 1]. That is float32 rounding, not a
different algorithm.

`align_corners=False` is load-bearing rather than decorative — `True` is also
bilinear, also plausible, and shifts every sample. Two tests pin it: one
against a longhand half-pixel reference implementation, one asserting the two
conventions really do differ.

## captum 0.9.0, not 0.8.0

`captum==0.8.0` — what upstream's unpinned conda environment resolves to —
requires `numpy<2.0`. Since every other pin here wants numpy 2.x, that is
direct evidence that upstream runs this entire stack on numpy 1.x. 0.9.0 has
the same `LayerGradCam` and no numpy cap.

## Tests

`tests/test_run.py`, **98 tests**, no GPU, no weights, no network: the network
and the GradCAM pass are stubbed, everything around them runs for real. The
tests that need the real checkpoints are marked `@pytest.mark.models` /
`@pytest.mark.gpu` and skipped by default; with `DOCSHAPEAXI_MODELS` set they
load all five.

## Verified through the API

Run against the real `condyles_4_class.ckpt` on an RTX 6000 Ada, through
`POST /run/DOCShapeAXI`, on a 294,260-point surface:

| | result |
|---|---|
| predictions only (`explain=false`) | **200**, 336 bytes, **9.8 s** |
| with the explanation | **200**, 11.5 MB archive, **42.1 s** |

The returned mesh carries **one array per class** —
`grad_cam_target_class_0` … `_3`, each in [0, 1] — beside the original
`PredictedID`. Upstream wrote its file inside the per-class loop to a
class-independent path, so it produced one array, four times over.

Two defects were found by that run and by nothing else:
`is_surface_file` received a `pathlib.Path` (the runner hands a tool `Path` for
its `path` arguments, while every unit test passed strings) and died on
`.lower()`; and `gradcam_process` reads `args.target_class`, without which
every class writes `grad_cam_max` and the last one wins. Both are pinned by
tests now.
