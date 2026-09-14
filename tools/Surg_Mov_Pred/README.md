# sadt-surgmovpred

Predicts the skeletal movements an orthognathic surgery will produce, from
pre-operative cephalometric measurements. One stacking regressor per predicted
measurement, each shipped with the scaler it was trained against.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, path
`SurgMovPred_CLI/SurgMovPred_CLI.py`, commit `d7702ae` (2026-06-24), by way of
`slicer-remote-tool-server`'s `tools/SurgMovPred/` -- whose history this
repository carries, so `git log --follow` on `src/sadt_surgmovpred/pipeline.py`
reaches back through it.

Upstream pins kept as-is: none are pinned upstream except `numpy==2.4.0`, which
was **not** adopted -- see "Versions" below.

Changes from upstream:

- **Model packages are loaded in sorted order** (`sorted(model.glob(...))`
  instead of `glob`'s filesystem order), so the output columns come out in the
  same order on every run and on every machine. No predicted value changes;
  only the column order does.
- **Both result tables are returned.** Upstream wrote `predictions_outputs.xlsx`
  and `predictions_outputs.csv` but returned only the Excel one, so the CSV
  never reached the client. `run()` returns `dict[str, Path]` with both.
- **Zip extraction and scratch directories removed.** The server unpacks
  archives before `run()` is called and hands the tool an output directory, so
  the tool no longer creates temporary folders, probes for a writable parent or
  extracts anything.
- **Logging is not configured by the tool.** Upstream attached its own stdout
  handler to a named logger; a library that configures logging takes the
  decision away from whatever runs it, so this uses a plain module logger and
  the runner owns the handlers.

The algorithm itself is unchanged: `clean_name`, `find_id_column`,
`predict_all_targets` and the feature-resolution rules are upstream's code.

## What it does

| | |
|---|---|
| Inputs | `measurements`: one CSV/XLSX/ODS table, one row per patient, or a folder of them for a batch. `model`: folder of model packages. `output_dir`: where results go. |
| Outputs | `predictions_outputs.xlsx` and `predictions_outputs.csv` -- one row per patient, one column per predicted measurement, plus `IDPatient`. |
| Model files | One subfolder per predicted measurement, each holding a `stacking_package.pkl` of `{target_name, features_names, scaler, model}`. Fetched by the server into `/DATA/SurgMovPred/models`; the shipped set is `all_models/` -- 112 packages, 1.4 GB. |
| GPU | None. This tool is CPU-only. |

Three behaviours worth knowing before reading a result:

- **A model whose features the input does not carry is skipped, not fatal.** A
  thinner table simply yields fewer predicted columns. Check the column count
  against the model count if a target you expected is missing.
- **The patient identifier is detected, not required.** `#`, `ID`, `PatientID`,
  `Patient Number`, `Subject_ID` and similar all match. With none of them,
  predictions still run and `IDPatient` comes out blank.
- **A batch folder is searched flat, not recursively.** The two discoveries in
  this tool differ on purpose: model packages are found with `**/`, because a
  bundle nests them one folder per target, while a batch of tables arrives as
  one unpacked archive. A folder holding only subfolders therefore yields no
  table at all and says so, rather than half a cohort.

## Versions

Every dependency is pinned to what the currently deployed server runs, because
that is the environment the reference predictions below were produced in:
numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, joblib 1.5.3, lightgbm 4.7.0,
openpyxl 3.1.5, odfpy 1.4.1.

- **`lightgbm` is not optional.** The stacking regressors hold `LGBMRegressor`
  sub-estimators, so `joblib.load` fails outright without it. It is absent from
  the tool's import list because nothing imports it by name.
- **The models were pickled under scikit-learn 1.6.1 and run under 1.7.2.** That
  mismatch is upstream's own design -- it silences `InconsistentVersionWarning`
  because compatibility is checked before a model ships -- and moving the pin to
  1.6.1 would be a change, not a fix.
- **Upstream's `numpy==2.4.0` was not adopted.** The deployed server runs 2.2.6,
  and 2.2.6 is what produced every number this port was checked against.
  Upstream pins 2.4.0 because Slicer's shared interpreter needs it for other
  reasons; here nothing else shares the environment. Moving to 2.4.0 is a
  decision for whoever revalidates against reference data.
- **`scipy` is deliberately unpinned.** The reference ran under 1.15.3 and this
  package resolves 1.18.0; predictions are bit-identical across the two (the
  Gaussian-process sub-estimators are the only scipy-sensitive part), so there
  is nothing to pin it to.

`requires-python` is `>=3.12,<3.13`. The reference was produced under Python
3.10 and this package runs 3.12 with bit-identical results, so the interpreter
was chosen for support life rather than to reproduce a number.

## Validated against

- **Input**: `DATA/SurgMovPred/testfiles/TestFiles/patients_to_predict.xlsx`,
  101 patients.
- **Weights**: `DATA/SurgMovPred/models/all_models`, 112 packages.
- **Reference**: the pre-port implementation
  (`slicer-remote-tool-server`, `tools/SurgMovPred/src/SurgMovPredLogic.py`) run
  on the same input and weights in the server's own environment
  (Python 3.10.12, scipy 1.15.3).
- **Result**: **bit-identical**. All 112 targets predicted for all 101 patients;
  max absolute difference 0.000e+00 across the 11 312 predicted values, and the
  `IDPatient` column matches exactly. Column *order* differs by design (see
  Changes from upstream); the column sets are equal.
- **Tolerance**: none needed -- equality was exact, so any future difference is a
  regression rather than noise.

GPU tests: none exist, the tool is CPU-only -- `torch` is not installed in this
venv at all, and a test asserts it.

The `models` marker is **deselected by default** rather than skipped
(`addopts = "-m 'not gpu and not models'"`): a plain `pytest` must not depend on
1.4 GB of packages being staged, and a green line that is half skips says
nothing.

## Working on it

```bash
cd tools/Surg_Mov_Pred
uv sync
uv run pytest              # 154 tests on models built in the test itself
uv run pytest -m models -o addopts=   # the real packages, see tests/data/README.md
```

```bash
# The schema the server publishes
.venv/bin/python ../../scripts/describe.py .
```
