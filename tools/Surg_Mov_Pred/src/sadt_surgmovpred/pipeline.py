"""Surgical movement prediction: one stacking regressor per target measurement.

Ported from the server-side tool with the algorithm untouched -- `clean_name`,
`find_id_column` and `predict_all_targets` are the upstream code. What is gone
is the plumbing the tool no longer owns: zip extraction, scratch directories and
`file_utils`. The server unpacks archives before `run()` is called and hands it
an output directory, so none of that belongs here any more.

`load_tabular_file` and `load_tabular_directory` are copied from the server's
`file_utils.py` rather than shared: see CONTRIBUTING.md on why there is no
sadt-core package.
"""

import logging
import re
from pathlib import Path

# No handler and no level: a library that configures logging steals the
# decision from whatever runs it. The runner owns the handlers.
logger = logging.getLogger(__name__)

TABULAR_SUFFIXES = (".csv", ".xlsx", ".ods")

MODEL_FILENAME = "stacking_package.pkl"


def clean_name(name: str) -> str:
    """Cleans a column name so it exactly matches the training-time format."""
    original = name
    try:
        name = str(name).strip()
        name = re.sub(r'[\r\n\t]', ' ', name)

        # 1. Strip problematic quotes and brackets
        name = name.replace('"', '').replace('\\', '').replace('[', '').replace(']', '')

        # 2. Replace spaces and dashes with underscores
        # (keeping the apostrophe (') for Jarabak's and SM_A'_CP!)
        name = re.sub(r'[^0-9a-zA-Z_\']+', '_', name)

        # 3. Collapse multiple underscores
        name = re.sub(r'_+', '_', name).strip('_')

        # 4. Model-specific adjustment
        name = name.replace("total", "Total")

        if re.match(r'^\d', name):
            name = f'f_{name}'

        return name or 'f_unnamed'
    except Exception as exc:
        # NOT a fallback name. Two columns that both failed here became the
        # same `f_unnamed`, so one silently shadowed the other and feature
        # resolution then matched the wrong measurement or none -- a wrong
        # prediction rather than a missing one, which is the worse of the two
        # for a clinician.
        raise ValueError(
            f"Could not derive a feature name from the column {original!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


# Common naming conventions used by different users for the patient identifier column.
# Matched case-insensitively, with optional separator (space/underscore/dash) between words.
ID_COLUMN_PATTERNS = [
    r'^#$',
    r'^id$',
    r'^id[\s_-]?patient$',
    r'^patient[\s_-]?id$',
    r'^patient[\s_-]?(num|number|no)$',
    r'^subject[\s_-]?id$',
    r'^patient$',
    r'^subject$',
]


def find_id_column(columns) -> str:
    """
    Tries to identify which input column holds the patient identifier.
    Naming conventions vary a lot between users (e.g. '#', 'ID', 'PatientID', 'Patient Number'...),
    so this matches a broad set of common patterns instead of a single fixed name.

    Returns the matching column name, or None if nothing matched.
    """
    normalized = [(col, str(col).strip().lower()) for col in columns]

    for pattern in ID_COLUMN_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE)
        for col, norm in normalized:
            if regex.fullmatch(norm):
                return col

    # Fallback: any column whose name contains both "patient" and "id"
    for col, norm in normalized:
        if 'patient' in norm and 'id' in norm:
            return col

    return None


def silence_sklearn_version_warning():
    """The shipped models are loaded by a different sklearn than trained them.

    That is expected -- compatibility is checked before a model ships -- so the
    warning is noise here, not a failure. It is silenced inside the pipeline
    rather than at import time so that importing this module stays free of
    sklearn.
    """
    import warnings

    try:
        from sklearn.exceptions import InconsistentVersionWarning
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    except ImportError:
        pass


def load_tabular_file(path: Path):
    """Load a single CSV, XLSX, or ODS file into a DataFrame."""
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".ods":
        return pd.read_excel(path, engine="odf")
    raise ValueError(f"Unsupported file extension '{suffix}' for tabular file: {path}")


def load_measurements(measurements: Path):
    """One table, or a folder of them concatenated into a batch.

    A folder is the batch form: the server pays a process start-up cost per
    call, so 40 clinics' spreadsheets have to arrive as one call. Sorted,
    because readdir order varies between filesystems and an unsorted
    concatenation renumbers every row between two runs on the same folder.
    """
    if measurements.is_dir():
        import pandas as pd

        files = sorted(
            path
            for path in measurements.iterdir()
            if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(
                f"No CSV, XLSX or ODS file found in: {measurements}"
            )
        # The count, never the path: `measurements` is the caller's own
        # folder, unpacked under its own name, so it can carry a patient's.
        logger.info(f"Loading {len(files)} table(s) from the input folder")
        return pd.concat([load_tabular_file(path) for path in files], ignore_index=True)

    if not measurements.is_file():
        raise FileNotFoundError(f"Input path does not exist: {measurements}")
    return load_tabular_file(measurements)


def load_model_packages(model: Path) -> dict:
    """Load every `stacking_package.pkl` under the model folder, keyed by target.

    One folder per predicted measurement, each holding one package of
    {target_name, features_names, scaler, model}.
    """
    import joblib

    if not model.is_dir():
        raise FileNotFoundError(f"Model path is not a folder: {model}")

    package_files = sorted(model.glob(f"**/{MODEL_FILENAME}"))
    if not package_files:
        raise FileNotFoundError(f"No '{MODEL_FILENAME}' model package found in {model}")

    logger.info(f"Loading {len(package_files)} model(s)...")
    packages = {}
    for pkl_path in package_files:
        try:
            package = joblib.load(pkl_path)
            packages[package['target_name']] = package
        except Exception:
            # One unreadable package must not lose the other 111: it is logged
            # and the run continues, exactly as upstream.
            logger.exception(f"Unable to load package {pkl_path}")

    if not packages:
        raise RuntimeError(
            f"None of the {len(package_files)} model package(s) found in {model} "
            "could be loaded."
        )

    logger.info(f"Successfully loaded {len(packages)} model package(s).")
    return packages


def predict_all_targets(df, packages: dict):
    """
    Predicts the values for each loaded model, dynamically adapting to the input data.
    """
    import pandas as pd

    try:
        # 1. Clean the input file's column names so they match the training-time format
        df_cleaned = df.copy()
        df_cleaned.columns = [clean_name(col) for col in df_cleaned.columns]

        # Collect predictions in a plain dict and build the DataFrame once at the end,
        # instead of inserting one column at a time (which fragments the DataFrame
        # and triggers pandas' PerformanceWarning).
        predictions_by_target = {}

        logger.info("Starting predictions for all target variables...")

        for target_name, pack in packages.items():
            try:
                expected_features = pack['features_names']
                scaler = pack['scaler']
                model = pack['model']

                # Resolve each expected feature to an actual column in the input file.
                # Some files provide T0 measurements without the "_T0" suffix used at training time.
                feature_source = {}
                missing_features = []
                for f in expected_features:
                    if f in df_cleaned.columns:
                        feature_source[f] = f
                    elif f.endswith('_T0') and f[:-3] in df_cleaned.columns:
                        feature_source[f] = f[:-3]
                    else:
                        missing_features.append(f)

                if missing_features:
                    logger.warning(f"⚠️ Model '{target_name}' skipped: missing {len(missing_features)} input feature(s) (e.g. {missing_features[:3]})")
                    continue

                # Extract and order the data according to this model's specific needs
                X_target = df_cleaned[[feature_source[f] for f in expected_features]]
                X_target.columns = expected_features

                # Standardize using the model's own scaler
                X_scaled = scaler.transform(X_target)
                X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features, index=df_cleaned.index)

                # Predict
                predictions_by_target[target_name] = model.predict(X_scaled_df)

            except Exception:
                logger.exception(f"Error predicting target '{target_name}'")

        if not predictions_by_target:
            # The loading stage already refuses to continue when NO package
            # could be loaded; this is the same guard one stage later, and the
            # missing half of it. Loading every model and predicting nothing
            # still wrote predictions_outputs.xlsx and .csv -- real files, with
            # the patient index and not one prediction column -- and reported
            # success. A guard has to count what the tool claims to have
            # produced, not the objects it walked past.
            raise RuntimeError(
                f"None of the {len(packages)} loaded model(s) produced a prediction. "
                "The measurements file is most likely missing the features they were "
                "trained on; the per-target errors are in the log."
            )

        results_df = pd.DataFrame(predictions_by_target, index=df_cleaned.index)

        logger.info(f"Predictions complete. {len(results_df.columns)} target(s) predicted out of {len(packages)} available.")
        return results_df

    except Exception:
        logger.exception("General error during prediction")
        raise


def save_results(df, output_dir: Path) -> dict:
    """Write the predictions table as both Excel and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_output = output_dir / "predictions_outputs.xlsx"
    csv_output = output_dir / "predictions_outputs.csv"

    logger.info(f"Saving results to: {output_dir}")
    df.to_excel(excel_output, index=True)
    df.to_csv(csv_output, index=True)

    logger.info("Results saved successfully!")
    return {"excel": excel_output, "csv": csv_output}


def predict(measurements: Path, model: Path, output_dir: Path) -> dict:
    import pandas as pd

    silence_sklearn_version_warning()
    logger.info("=== Surgical Movements Prediction Engine (Stacking Deploy) ===")

    packages = load_model_packages(model)

    df_input = load_measurements(measurements)
    logger.info(f"Input data loaded: {len(df_input)} rows")

    # Recover the patient identifier from the input so it can be carried over to
    # the output: a table of 101 unlabelled prediction rows is unusable.
    id_column = find_id_column(df_input.columns)
    if id_column is not None:
        logger.info(f"Detected patient ID column: '{id_column}'")
        id_values = df_input[id_column].reset_index(drop=True)
    else:
        logger.warning(
            "Could not detect a patient ID column in the input data; 'IDPatient' "
            "will be left empty in the output."
        )
        id_values = pd.Series([pd.NA] * len(df_input))

    df_results = predict_all_targets(df_input, packages)
    df_results.insert(0, 'IDPatient', id_values.values)

    outputs = save_results(df_results, output_dir)
    logger.info("=== Process completed successfully ===")
    return outputs
