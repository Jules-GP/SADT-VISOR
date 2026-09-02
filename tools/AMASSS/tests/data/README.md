# Test data

Nothing is committed here. The model bundle is 2.1 GB and `MG_test_scan.nii.gz`
is a real CBCT -- patient data, which never goes into this repository.

Everything the CI suite needs, it builds: `test_run.py` writes 8³ synthetic
volumes and stubs `nnunet_runner.predict_folder`, so input discovery, model
resolution, format conversion, label merging, file naming and the report are all
exercised without a checkpoint or a GPU.

## Running the real-model test

`test_real_models_segment_a_real_scan` is marked `gpu` and `models` and skips
unless both paths are given:

```bash
SADT_AMASSS_MODELS=/path/to/DATA/AMASSS/models/AMASSS_Models \
SADT_AMASSS_SCAN=/path/to/DATA/AMASSS/testfiles/MG_test_scan.nii.gz \
uv run pytest -m models
```

Both live under the deployment's `/DATA` mount, or under `DATA/AMASSS/` in a
`VISOR-serve` checkout. It needs a CUDA device and takes a few
minutes for three structures. Write manual runs into the repository's gitignored
`output/` directory, never next to the inputs.

If an anonymised, publishable CBCT of the right geometry ever exists, it belongs
here as a download script plus checksums. Ask before committing anything.
