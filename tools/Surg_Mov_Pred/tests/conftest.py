"""Fixtures shared by the SurgMovPred suites.

Everything here builds a real scikit-learn package on the fly: the shipped
models are 1.4 GB and the reference table is patient data, so neither can be
committed. A package built here has the same four keys the real ones do
(`target_name`, `features_names`, `scaler`, `model`), which is the whole of the
contract `load_model_packages` and `predict_all_targets` rely on.
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def build_package(features, coefficients=None, samples=None):
    """A package with the shape the real ones have, over `features`.

    Fitted on a NAMED frame, like the real packages: a scaler fitted on a bare
    array warns on every transform and buries anything else in the log.
    """
    samples = samples or [
        [value * (index + 1) for index in range(len(features))]
        for value in (0, 10, 20)
    ]
    training = pd.DataFrame(samples, columns=list(features))
    scaler = StandardScaler().fit(training)
    scaled = pd.DataFrame(scaler.transform(training), columns=training.columns)
    targets = coefficients or [5, 15, 25]
    model = LinearRegression().fit(scaled, targets)
    return {
        "target_name": None,  # filled in by write_package
        "features_names": list(features),
        "scaler": scaler,
        "model": model,
    }


def write_package(folder: Path, target_name: str, features=("f1",), subdir=None) -> Path:
    """Write one `stacking_package.pkl` under `folder/<subdir or target>/`."""
    package = build_package(features)
    package["target_name"] = target_name
    target_dir = folder / (subdir or target_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, target_dir / "stacking_package.pkl")
    return folder


class RecordingModel:
    """Stands in for a stacking regressor and keeps what it was handed.

    The prediction path is the one thing a repackaging could silently break --
    a column reordered, an index dropped, an unscaled frame -- so it is
    asserted on the object the model actually receives rather than on the
    number that comes out the far end.
    """

    def __init__(self, value=0.0):
        self.seen = []
        self.value = value

    def predict(self, frame):
        self.seen.append(frame.copy())
        return [self.value + index for index in range(len(frame))]


class RecordingScaler:
    """A scaler that records its input and passes it through unchanged."""

    def __init__(self):
        self.seen = []

    def transform(self, frame):
        self.seen.append(frame.copy())
        return frame.to_numpy(dtype=float)


@pytest.fixture
def model_folder(tmp_path):
    """One model, `f1_Pred`, over the single feature `f1`."""
    return write_package(tmp_path / "models", "f1_Pred")


@pytest.fixture
def patients_table(tmp_path):
    """Three patients, an identifier column, and the one feature `f1`."""
    table = tmp_path / "patients.xlsx"
    pd.DataFrame({"PatientID": [1, 2, 3], "f1": [0, 10, 20]}).to_excel(table, index=False)
    return table
