# Test data

Nothing is committed here, and there is no download script, because neither
fixture can be published:

- the model packages are 1.4 GB (112 × ~13 MB), well past what belongs in git;
- `patients_to_predict.xlsx` is a table of real cephalometric measurements for
  101 patients -- patient data, which never goes into this repository.

The tests that need neither build their own scikit-learn models in
`test_run.py`, which is why the suite runs in CI without any of this.

## Running the real-model test

`test_real_models_predict_every_target` is marked `models`, which is
**deselected by default** -- `-o addopts=` turns the default selection off so
`-m models` can select it -- and skips unless both paths are given. Point them
at the server's data store:

```bash
SADT_SURGMOVPRED_MODELS=/path/to/DATA/SurgMovPred/models/all_models \
SADT_SURGMOVPRED_INPUT=/path/to/DATA/SurgMovPred/testfiles/TestFiles/patients_to_predict.xlsx \
uv run pytest -m models -o addopts=
```

It runs in about ten seconds: 112 packages loaded, 112 targets predicted for
101 patients.

Both live under the deployment's `/DATA` mount, or under `DATA/SurgMovPred/` in
a `VISOR-serve` checkout. Write results to the repository's
gitignored `output/` directory if you run the tool by hand -- never next to the
inputs.

If a public, anonymised sample of the input format ever exists, it belongs here
as a download script plus checksums. Ask before committing anything.
