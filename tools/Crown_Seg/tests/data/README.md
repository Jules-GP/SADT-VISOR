# Test data

Nothing is committed here. The CI suite builds its own three-triangle meshes
with VTK and stubs `_run_shapeaxi`, so input discovery, the already-segmented
bypass, the output tree and the run report are all exercised without a
checkpoint, a GPU, or the `segmentation` extra.

## Staging the checkpoint and a mesh

Both are small and public, and the server's `scripts/data-manifest.yml` holds
the URLs and sizes:

| File | Size | Where |
|---|---|---|
| `07-21-22_val-loss0.169.pth` | 6.6 MB | Fly-by-CNN release 3.0 |
| `T1_01_U_segmented.vtk` | 15.9 MB | ALIDDM release v1.0.4 |

```bash
scripts/setup-models.sh --tool CrownSeg      # from a VISOR-serve checkout
```

Note that the published test mesh is **already segmented** -- it carries a
`Universal_ID` array. It exercises the passthrough branch as it is; pass
`skip_segmented=False` to force the network to run on it anyway, which is what
the real-model test does.

## Running the real-model test

`test_real_model_labels_a_real_mesh` is marked `gpu` and `models`, and needs
the engine, which a plain `uv sync` does not install:

```bash
uv sync --extra segmentation     # builds pytorch3d from source, see README.md
SADT_CROWNSEG_MODEL=/path/to/DATA/CrownSeg/models/07-21-22_val-loss0.169.pth \
SADT_CROWNSEG_MESH=/path/to/DATA/CrownSeg/testfiles/T1_01_U_segmented.vtk \
uv run pytest -m models
```

Write manual runs into the repository's gitignored `output/` directory, never
next to the inputs.
