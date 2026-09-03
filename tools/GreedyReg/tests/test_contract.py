"""Every claim README.md makes, and the shape the server reads this tool through.

The schema the client builds its panel from is generated from `run()`'s
signature and docstring, so both are part of the contract rather than internal
detail -- and so are the things the README says were deliberately NOT ported,
which nothing else would notice creeping back in.
"""

import ast
import inspect
import re
import tomllib
import typing
from pathlib import Path

import pytest

import sadt_greedyreg
from sadt_greedyreg import pipeline

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
SOURCES = {path.name: path.read_text() for path in (ROOT / "src").rglob("*.py")}
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())

SIGNATURE = inspect.signature(sadt_greedyreg.run)


def top_level_imports(source: str) -> set:
    """Module names imported at column zero, i.e. at import time."""
    found = set()
    for line in source.splitlines():
        match = re.match(r"(?:import|from) ([A-Za-z_][\w.]*)", line)
        if match:
            found.add(match.group(1).split(".")[0])
    return found


# ---------------------------------------------------------------------------
# "Greedy is a Python package, not a binary to find"
# ---------------------------------------------------------------------------

def test_greedy_is_a_pinned_dependency_rather_than_an_executable():
    """Upstream took the greedy executable's PATH as an argument and made the
    Slicer module find it on the user's machine. A client cannot supply a path
    on the server."""
    pinned = [dep for dep in PYPROJECT["project"]["dependencies"]
              if dep.startswith("picsl_greedy")]
    assert pinned == ["picsl_greedy==1.4.0.1"]


def test_no_argument_asks_the_caller_where_greedy_is():
    for name in SIGNATURE.parameters:
        assert "binary" not in name.lower()
        assert "greedy" not in name.lower()


def test_the_source_names_no_greedy_executable():
    for name, source in SOURCES.items():
        assert "greedyBinary" not in source, name
        assert "greedy_binary" not in source, name


def test_greedy_is_reached_through_its_own_python_distribution():
    assert "from picsl_greedy import Greedy3D" in pipeline._CHILD


# ---------------------------------------------------------------------------
# "The patient rule is the shared one"
# ---------------------------------------------------------------------------

def test_identity_comes_from_the_shared_package():
    """CONTRIBUTING.md lists identity derivation as one of the three things
    that belong in a shared package: two tools exchanging files by name have
    to agree on where the identifier ends."""
    assert "sadt_areg_common" in top_level_imports(SOURCES["__init__.py"])
    assert "sadt-areg-common" in PYPROJECT["project"]["dependencies"]


def test_this_tool_writes_no_patient_regex_of_its_own():
    """Upstream's was `^([A-Za-z]+\\d+)`; a second copy here would be a second
    answer to the question the shared package exists to answer once."""
    for name, source in SOURCES.items():
        assert "re" not in top_level_imports(source), name
        assert "re.compile" not in source, name
        assert "re.match" not in source, name


# ---------------------------------------------------------------------------
# "One patient failing costs one patient"
# ---------------------------------------------------------------------------

def test_nothing_in_this_tool_exits_the_process():
    """Upstream's per-patient `except` called `sys.exit(1)`, which on a server
    would end the run for every patient behind it as well. Read from the syntax
    tree, so the comment recording that history does not count as a call."""
    for name, source in SOURCES.items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                assert ast.unparse(node.func) not in ("sys.exit", "exit", "quit"), name


# ---------------------------------------------------------------------------
# "Not ported"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("marker", ["<filter-progress>", "<filter-comment>"])
def test_the_slicer_progress_protocol_is_not_ported(marker):
    """It is how the Qt panel drove its progress bar, by parsing the CLI's
    stdout. There is no panel here, and stdout is the child's."""
    for name, source in SOURCES.items():
        assert marker not in source, name


def test_there_is_no_dependency_install_prompt():
    """Upstream guarded its nibabel import with a message telling the user to
    open the module and let it install the package. A server installs nothing
    mid-request."""
    for name, source in SOURCES.items():
        assert "pip install" not in source, name


def test_the_heavy_imports_are_lazy():
    """`nibabel` and `numpy` are imported inside the two functions that use
    them, so importing this package -- which is what reading its schema does --
    costs nothing."""
    imported = top_level_imports(SOURCES["pipeline.py"]) | top_level_imports(SOURCES["__init__.py"])
    assert "nibabel" not in imported
    assert "numpy" not in imported


def test_the_registration_tuning_is_not_exposed():
    """The multi-resolution schedule and the random search describe how this
    registration was tuned, not a per-request choice."""
    for name in SIGNATURE.parameters:
        assert not any(word in name.lower() for word in ("search", "schedule", "iteration"))
    assert "100x100x50x25" not in str(SIGNATURE)


def test_the_readme_quotes_the_schedule_the_command_carries():
    """README: "the affine search with its `-n 100x100x50x25` schedule". A
    document quoting a flag the command no longer passes is worse than one that
    says nothing."""
    assert "100x100x50x25" in README
    command = pipeline.registration_command("f", "m", "o", "i", "NCC", "Rigid")
    assert command[command.index("-n") + 1] == "100x100x50x25"


def test_the_readme_quotes_the_random_search_the_command_carries():
    assert "-search 100 10 20" in README
    command = pipeline.registration_command("f", "m", "o", "i", "NCC", "Rigid")
    index = command.index("-search")
    assert command[index:index + 4] == ["-search", "100", "10", "20"]


# ---------------------------------------------------------------------------
# The schema the server publishes
# ---------------------------------------------------------------------------

def test_run_is_the_entry_point_the_runner_imports():
    assert sadt_greedyreg.__all__ == ["run"]
    assert callable(sadt_greedyreg.run)


def test_the_arguments_are_the_documented_ones():
    assert list(SIGNATURE.parameters) == [
        "t1", "t2", "output_dir", "masks", "initial_transforms",
        "metric", "transform_type", "output_suffix"]


def test_every_argument_is_described_in_the_docstring():
    """The published description of each argument is read from here, so an
    undocumented argument reaches the panel as a bare name."""
    documentation = sadt_greedyreg.run.__doc__
    for name in SIGNATURE.parameters:
        assert f"{name}:" in documentation, name


def test_the_three_folders_are_required_or_optional_as_documented():
    parameters = SIGNATURE.parameters
    assert parameters["t1"].default is inspect.Parameter.empty
    assert parameters["t2"].default is inspect.Parameter.empty
    assert parameters["output_dir"].default is inspect.Parameter.empty
    assert parameters["masks"].default == ""
    assert parameters["initial_transforms"].default == ""


def test_the_defaults_are_the_documented_ones():
    parameters = SIGNATURE.parameters
    assert parameters["metric"].default == "NCC"
    assert parameters["transform_type"].default == "Rigid"
    assert parameters["output_suffix"].default == "registered"


def test_the_published_choices_are_exactly_what_the_pipeline_accepts():
    """The schema's `choices` come from these annotations and the refusal comes
    from the pipeline's tables: if they disagree, the panel offers a value the
    tool answers 422 for."""
    hints = typing.get_type_hints(sadt_greedyreg.run)
    assert typing.get_args(hints["metric"]) == pipeline.METRICS
    assert typing.get_args(hints["transform_type"]) == pipeline.TRANSFORMS


def test_the_tool_declares_no_server_side_data():
    """README: no data is declared for this tool. The server fills an argument
    named `model`, `*_model` or `*_reference` from `DATA/<tool>/models/`, so
    naming one that way is how a tool asks for a bundle."""
    for name in SIGNATURE.parameters:
        assert name != "model"
        assert not name.endswith("_model")
        assert not name.endswith("_reference")


def test_the_output_directory_argument_exists_for_the_server_to_fill():
    """It is taken out of the published schema and filled in at dispatch with
    the job's own `output/`; a tool that does not declare it is handed nowhere
    to write."""
    assert "output_dir" in SIGNATURE.parameters


def test_the_package_is_registered_under_the_folder_name():
    """`dispatch.py` looks the interpreter up at
    `<TOOLS_DIR>/<tool name>/.venv/bin/python`, so a name that is not the
    folder's registers a tool that cannot be run."""
    assert PYPROJECT["tool"]["sadt"]["name"] == ROOT.name == "GreedyReg"
    assert PYPROJECT["tool"]["sadt"]["tool"] is True


def test_the_report_the_docstring_promises_is_the_one_that_is_written():
    assert "GreedyReg_report.json" in sadt_greedyreg.run.__doc__
    assert "GreedyReg_report.json" in SOURCES["__init__.py"]


# ---------------------------------------------------------------------------
# How the suite is meant to be run
# ---------------------------------------------------------------------------

def test_the_markers_the_readme_deselects_are_registered():
    """`uv run pytest -m "not gpu"` only means anything if the marker exists;
    an unregistered one is a warning and a filter that matches nothing."""
    markers = PYPROJECT["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.startswith("gpu:") for marker in markers)
    assert any(marker.startswith("models:") for marker in markers)


def test_no_test_in_this_suite_needs_a_gpu_or_real_scans():
    """Everything here runs on 4x4x4 volumes with greedy stubbed, so the suite
    is seconds rather than minutes and a machine with no card runs all of it.
    A test carrying either marker would be deselected by the README's own
    command, i.e. would not actually be run by anyone."""
    for path in (ROOT / "tests").glob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                applied = ast.unparse(decorator)
                assert not applied.startswith("pytest.mark.gpu"), f"{path.name}::{node.name}"
                assert not applied.startswith("pytest.mark.models"), f"{path.name}::{node.name}"
