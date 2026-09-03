"""The claims README.md makes, asserted rather than described.

A README records what was decided and why; each decision that nothing checks
quietly stops being true. These are the checkable ones: the pins the reference
predictions were produced under, the interpreter, the plumbing the port deleted,
and the schema the server publishes.
"""

import ast
import json
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pytest

import sadt_surgmovpred
from sadt_surgmovpred import run
from sadt_surgmovpred import pipeline

TOOL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_DIR.parents[1]


def pyproject():
    return tomllib.loads((TOOL_DIR / "pyproject.toml").read_text(encoding="utf-8"))


def readme():
    return (TOOL_DIR / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# "Every dependency is pinned to what the currently deployed server runs"
# ---------------------------------------------------------------------------

PINNED = {
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scikit-learn": "1.7.2",
    "joblib": "1.5.3",
    "lightgbm": "4.7.0",
    "openpyxl": "3.1.5",
    "odfpy": "1.4.1",
}


@pytest.mark.parametrize("package, expected", sorted(PINNED.items()))
def test_the_installed_version_is_the_one_the_reference_was_produced_under(
    package, expected
):
    """The bit-identical comparison in README was run in this exact set. A
    dependency that moved without the README moving makes that claim false."""
    assert version(package) == expected


@pytest.mark.parametrize("package, expected", sorted(PINNED.items()))
def test_pyproject_pins_exactly_that_version(package, expected):
    """The pin and the installed version are two different facts: a lock file
    can deliver something the pyproject never asked for."""
    dependencies = pyproject()["project"]["dependencies"]
    assert f"{package}=={expected}" in dependencies


@pytest.mark.parametrize("package, expected", sorted(PINNED.items()))
def test_the_readme_names_that_version(package, expected):
    """The Versions paragraph is what a maintainer reads before changing a
    pin; drift there is drift in the only record of the reference environment."""
    assert f"{package} {expected}" in readme()


def test_scipy_is_deliberately_unpinned():
    """The reference ran under 1.15.3 and this package resolves 1.18.0 with
    bit-identical predictions, so there is nothing to pin it to -- and pinning
    it would claim a precision the validation did not establish."""
    assert not any(
        dependency.lower().startswith("scipy")
        for dependency in pyproject()["project"]["dependencies"]
    )


def test_upstreams_numpy_pin_was_not_adopted():
    """Upstream pins numpy 2.4.0 for Slicer's shared interpreter. Nothing
    shares this environment, and 2.2.6 is what produced every reference number."""
    assert "numpy==2.4.0" not in pyproject()["project"]["dependencies"]


def test_the_interpreter_is_the_one_the_port_was_validated_on():
    assert pyproject()["project"]["requires-python"] == ">=3.12,<3.13"
    assert sys.version_info[:2] == (3, 12)


def imported_names(path: Path) -> set:
    """Every module name this file imports, at any nesting depth."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_lightgbm_is_a_dependency_although_no_import_statement_names_it(tmp_path):
    """The stacking regressors hold LGBMRegressor sub-estimators, so
    `joblib.load` fails outright without it -- and reading every import in this
    package would never tell you that. It is the easiest pin to delete by
    mistake, so both halves are asserted: absent from the source, present in
    the dependencies, and able to round-trip through joblib."""
    import joblib
    import lightgbm
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LinearRegression

    imported = set()
    for path in (TOOL_DIR / "src").rglob("*.py"):
        imported |= imported_names(path)
    assert "lightgbm" not in imported

    assert any(
        dependency.startswith("lightgbm")
        for dependency in pyproject()["project"]["dependencies"]
    )

    frame = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0]})
    stack = StackingRegressor(
        estimators=[("lgbm", lightgbm.LGBMRegressor(n_estimators=2, verbose=-1))],
        final_estimator=LinearRegression(),
        cv=2,
    ).fit(frame, [0.0, 1.0, 2.0, 3.0])

    pickled = tmp_path / "stacking_package.pkl"
    joblib.dump(stack, pickled)
    assert len(joblib.load(pickled).predict(frame)) == 4


def test_the_tool_is_cpu_only():
    """README: "GPU: None". Nothing in this environment can reach a card, which
    is why `MAX_CONCURRENT_GPU_JOBS` never has to admit a run of this tool."""
    import importlib.util

    assert importlib.util.find_spec("torch") is None
    assert importlib.util.find_spec("nvidia") is None


# ---------------------------------------------------------------------------
# "Zip extraction and scratch directories removed"
# ---------------------------------------------------------------------------

def test_the_package_creates_no_temporary_directory_and_unpacks_nothing():
    """The server unpacks archives before `run()` and hands the tool an output
    directory; a tool doing either again would be doing it twice."""
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (TOOL_DIR / "src").rglob("*.py")
    )
    for absent in ("zipfile", "tempfile", "mkdtemp", "shutil", "extractall"):
        assert absent not in source, absent


# ---------------------------------------------------------------------------
# "Logging is not configured by the tool"
# ---------------------------------------------------------------------------

def test_the_tool_attaches_no_handler_and_sets_no_level():
    """Upstream attached its own stdout handler to a named logger. A library
    that configures logging takes the decision away from whatever runs it."""
    import logging

    assert pipeline.logger.handlers == []
    assert pipeline.logger.level == logging.NOTSET
    assert pipeline.logger.propagate is True


def test_importing_the_package_touches_no_root_handler():
    import logging

    before = list(logging.getLogger().handlers)
    import importlib

    importlib.reload(pipeline)
    assert logging.getLogger().handlers == before


def test_a_run_prints_nothing_to_stdout(tmp_path, model_folder, patients_table, capsys):
    """Everything the tool has to say goes through the logger, so the runner
    decides where it lands -- and a patient's file name never lands on stdout."""
    run(measurements=patients_table, model=model_folder, output_dir=tmp_path / "out")

    assert capsys.readouterr().out == ""


def test_the_log_never_carries_a_measurement_value(tmp_path, model_folder, caplog):
    """Logs are limited to what a run did, never to what was in it."""
    import logging

    table = tmp_path / "patients.csv"
    pd.DataFrame({"PatientID": [1], "f1": [1234.5678]}).to_csv(table, index=False)

    with caplog.at_level(logging.DEBUG):
        run(measurements=table, model=model_folder, output_dir=tmp_path / "out")

    assert "1234.5678" not in caplog.text


# ---------------------------------------------------------------------------
# The package shape the models must have
# ---------------------------------------------------------------------------

def test_a_package_missing_its_target_name_is_dropped_not_loaded_under_a_wrong_key(
    tmp_path, caplog
):
    """`packages[package['target_name']]` is the only place the key comes from;
    a package without one must not silently become a column named None."""
    import joblib

    from conftest import build_package

    package = build_package(["f1"])
    del package["target_name"]
    (tmp_path / "models" / "a").mkdir(parents=True)
    joblib.dump(package, tmp_path / "models" / "a" / "stacking_package.pkl")

    with pytest.raises(RuntimeError, match="could be loaded"):
        pipeline.load_model_packages(tmp_path / "models")


def test_a_package_missing_its_feature_list_costs_only_its_own_target():
    from conftest import build_package

    good = build_package(["f1"])
    broken = build_package(["f1"])
    del broken["features_names"]

    results = pipeline.predict_all_targets(
        pd.DataFrame({"f1": [0.0, 10.0]}), {"broken": broken, "good": good}
    )

    assert list(results.columns) == ["good"]


# ---------------------------------------------------------------------------
# The schema the server publishes
# ---------------------------------------------------------------------------

def test_the_declared_tool_name_is_the_api_identity():
    """`[tool.sadt] name` is what a client sends and what `deployment.toml` is
    keyed by. It is the folder name here, and changing either alone breaks the
    HTTP contract."""
    assert pyproject()["tool"]["sadt"]["name"] == TOOL_DIR.name == "Surg_Mov_Pred"
    assert pyproject()["tool"]["sadt"]["tool"] is True


def test_every_argument_of_run_is_documented_for_the_client():
    """`describe.py` reads each argument's description out of `run()`'s
    docstring and refuses to emit a schema without one; the client renders it
    under the field."""
    import inspect

    arguments = set(inspect.signature(run).parameters)
    documented = run.__doc__.split("Args:", 1)[1]
    for name in arguments:
        assert f"{name}:" in documented, name


def test_describe_publishes_the_schema_this_tool_promises():
    """The signature is the single source of truth for the panel a clinician
    sees. Running the real generator is the only way to know it still holds."""
    script = REPO_ROOT / "scripts" / "describe.py"
    if not script.is_file():
        pytest.skip("run from a full SADT-VISOR checkout")

    finished = subprocess.run(
        [sys.executable, str(script), str(TOOL_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr

    schema = json.loads(finished.stdout)
    assert schema["name"] == "Surg_Mov_Pred"
    assert sorted(schema["arguments"]) == ["measurements", "model", "output_dir"]
    assert all(argument["type"] == "path" for argument in schema["arguments"].values())


def test_the_public_surface_is_run_alone():
    """README: "Only `run` is public." Everything else is `pipeline`, which the
    server never imports."""
    tree = ast.parse((TOOL_DIR / "src" / "sadt_surgmovpred" / "__init__.py").read_text())
    defined = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert defined == {"run"}
    assert sadt_surgmovpred.run is run
