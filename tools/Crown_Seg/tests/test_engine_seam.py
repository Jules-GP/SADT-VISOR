"""The seam between this tool and shapeaxi.

Nothing here is ported: the Slicer modules ran the `dentalmodelseg` executable
out of Slicer's own bin directory, and that executable is only the console
script of `shapeaxi.dental_model_seg`. There is no binary to shell out to, so
`main()` is called directly with the namespace its own `cml()` would have
built -- which makes that namespace the contract, and the thing to pin.
"""

import sys
import types

import pytest

from sadt_crownseg import pipeline
from sadt_crownseg.errors import ToolInputError, ToolUnavailableError


@pytest.fixture
def captured_namespace(monkeypatch):
    """Run `_run_shapeaxi` against a stand-in `main()` and keep what it got."""
    seen = {}

    class FakeModule:
        @staticmethod
        def main(args):
            seen.update(vars(args))

    monkeypatch.setattr(pipeline, "_import_dental_model_seg", lambda: FakeModule)
    return seen


def invoke(tmp_path, **overrides):
    arguments = dict(
        csv_path=str(tmp_path / "crownseg_input.csv"),
        output_dir=str(tmp_path / "out"),
        model_path=str(tmp_path / "model.pth"),
        input_root=str(tmp_path / "cohort"),
        array_name="Universal_ID",
        suffix="Seg",
        device="cpu",
        fdi=False,
        num_workers=2,
    )
    arguments.update(overrides)
    pipeline._run_shapeaxi(**arguments)


# ---------------------------------------------------------------------------
# The namespace cml() would have built
# ---------------------------------------------------------------------------

def test_the_namespace_carries_exactly_the_fields_shapeaxi_reads(tmp_path, captured_namespace):
    """An extra field is harmless; a MISSING one is an AttributeError inside
    shapeaxi's `main()`, after the model has loaded."""
    invoke(tmp_path)

    assert set(captured_namespace) == {
        "surf", "csv", "model", "suffix", "out", "num_workers",
        "crown_segmentation", "array_name", "fdi", "overwrite", "device",
        "vtk_folder",
    }


def test_the_csv_branch_is_selected_by_leaving_surf_unset(tmp_path, captured_namespace):
    """`surf` and `csv` are shapeaxi's two input modes. Setting both would make
    it segment one mesh and ignore the batch."""
    invoke(tmp_path)

    assert captured_namespace["surf"] is None
    assert captured_namespace["csv"] == str(tmp_path / "crownseg_input.csv")


def test_the_output_root_and_the_tree_root_are_two_different_arguments(
    tmp_path, captured_namespace
):
    """`out` is where results go, `vtk_folder` is the prefix stripped off each
    input path to rebuild the tree under it. Swapping them flattens a cohort."""
    invoke(tmp_path)

    assert captured_namespace["out"] == str(tmp_path / "out")
    assert captured_namespace["vtk_folder"] == str(tmp_path / "cohort")


def test_the_crown_segmentation_and_overwrite_flags_stay_off(tmp_path, captured_namespace):
    """`crown_segmentation` is shapeaxi's own post-processing pass and
    `overwrite` would let a second run replace the first's outputs in place;
    neither is what this tool asks for."""
    invoke(tmp_path)

    assert captured_namespace["crown_segmentation"] == 0
    assert captured_namespace["overwrite"] == 0


def test_the_checkpoint_and_the_array_name_are_passed_through_verbatim(
    tmp_path, captured_namespace
):
    invoke(tmp_path, array_name="PredictedID", suffix="Pred", device="cuda:0")

    assert captured_namespace["model"] == str(tmp_path / "model.pth")
    assert captured_namespace["array_name"] == "PredictedID"
    assert captured_namespace["suffix"] == "Pred"
    assert captured_namespace["device"] == "cuda:0"


# ---------------------------------------------------------------------------
# What shapeaxi prints, and what it raises
# ---------------------------------------------------------------------------

def test_shapeaxis_stdout_never_reaches_the_process_output(tmp_path, monkeypatch, capsys):
    """It prints "Saving results to <path>" for every mesh, and that path
    carries the patient's own file name -- which must not reach a log."""
    class Chatty:
        @staticmethod
        def main(args):
            print("Saving results to /data/patients/Smith_John_T1_Seg.vtk")

    monkeypatch.setattr(pipeline, "_import_dental_model_seg", lambda: Chatty)

    invoke(tmp_path)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Smith_John" not in captured.err


def test_a_real_failure_from_shapeaxi_is_left_untouched(tmp_path, monkeypatch):
    """Swallowing stdout must not swallow the exception with it: the run has to
    fail, and the server has to see why."""
    class Failing:
        @staticmethod
        def main(args):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(pipeline, "_import_dental_model_seg", lambda: Failing)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        invoke(tmp_path)


def test_stdout_is_restored_after_the_call(tmp_path, monkeypatch, capsys):
    """A redirect left in place would silence the whole worker process."""
    class Quiet:
        @staticmethod
        def main(args):
            pass

    monkeypatch.setattr(pipeline, "_import_dental_model_seg", lambda: Quiet)

    invoke(tmp_path)
    print("still here")

    assert capsys.readouterr().out == "still here\n"


def test_stdout_is_restored_even_when_shapeaxi_raises(tmp_path, monkeypatch, capsys):
    class Failing:
        @staticmethod
        def main(args):
            print("half a mesh")
            raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "_import_dental_model_seg", lambda: Failing)

    with pytest.raises(RuntimeError):
        invoke(tmp_path)
    print("still here")

    assert capsys.readouterr().out == "still here\n"


# ---------------------------------------------------------------------------
# A venv without the segmentation extra
# ---------------------------------------------------------------------------

def test_an_absent_shapeaxi_says_how_to_install_it(monkeypatch):
    """`uv sync` alone cannot run the network, and has to say so. The message
    is the only thing between a maintainer and a bare ModuleNotFoundError from
    inside shapeaxi -- and a 501 body a clinician reads."""
    monkeypatch.setitem(sys.modules, "shapeaxi", None)

    with pytest.raises(ToolUnavailableError) as failure:
        pipeline._import_dental_model_seg()

    message = str(failure.value)
    assert "uv sync --extra segmentation" in message
    assert "tools/Crown_Seg" in message
    assert "missing: shapeaxi" in message


def test_the_engine_import_probe_runs_the_workaround(monkeypatch):
    """The import is the only place the workaround can be applied, because
    `dental_model_seg` reads the name at call time. Run against a stand-in
    shapeaxi so it holds in a venv without the extra, which is what CI has."""
    shapeaxi = types.ModuleType("shapeaxi")
    shapeaxi.dental_model_seg = types.ModuleType("shapeaxi.dental_model_seg")
    shapeaxi.saxi_nets = types.ModuleType("shapeaxi.saxi_nets")
    monkeypatch.setitem(sys.modules, "shapeaxi", shapeaxi)
    monkeypatch.setitem(sys.modules, "shapeaxi.dental_model_seg", shapeaxi.dental_model_seg)
    monkeypatch.setitem(sys.modules, "shapeaxi.saxi_nets", shapeaxi.saxi_nets)

    called = []
    monkeypatch.setattr(
        pipeline, "_restore_moved_class", lambda saxi_nets: called.append(saxi_nets)
    )

    assert pipeline._import_dental_model_seg() is shapeaxi.dental_model_seg
    assert called == [shapeaxi.saxi_nets]


def test_the_shapeaxi_version_is_readable_for_the_warning():
    """The workaround logs which release needed it, so the report at
    DCBIA-OrthoLab/ShapeAXI can name a version."""
    assert isinstance(pipeline._shapeaxi_version(), str)


# ---------------------------------------------------------------------------
# The shapeaxi 2.0.x workaround
# ---------------------------------------------------------------------------

class _MissingTheClass:
    """shapeaxi 2.0.x: `dental_model_seg` reads DentalModelSeg off this module,
    and it is not there."""


class _HasTheClass:
    DentalModelSeg = object()


@pytest.fixture
def moved_class(monkeypatch):
    """A `shapeaxi.saxi_nets_lightning` holding the class 2.0.x misplaced."""
    moved = types.ModuleType("shapeaxi.saxi_nets_lightning")
    moved.DentalModelSeg = object()
    monkeypatch.setitem(sys.modules, "shapeaxi.saxi_nets_lightning", moved)
    monkeypatch.setitem(sys.modules, "shapeaxi", types.ModuleType("shapeaxi"))
    return moved


def test_the_workaround_points_the_name_back_at_where_the_class_went(moved_class):
    """Without it, `dental_model_seg.main()` raises AttributeError before it
    touches a mesh -- and the pre-port tool fails identically on the deployed
    image."""
    broken = _MissingTheClass()

    assert pipeline._restore_moved_class(broken) is True
    assert broken.DentalModelSeg is moved_class.DentalModelSeg


def test_the_workaround_says_which_release_needed_it(moved_class, caplog):
    """It has to be reported upstream, so the log has to name a version."""
    import logging

    with caplog.at_level(logging.WARNING):
        pipeline._restore_moved_class(_MissingTheClass())

    assert "DentalModelSeg" in caplog.text
    assert "saxi_nets_lightning" in caplog.text


def test_the_workaround_is_a_no_op_once_upstream_puts_the_name_back():
    """The `hasattr` guard is what makes this delete itself rather than
    linger."""
    fixed = _HasTheClass()
    before = fixed.DentalModelSeg

    assert pipeline._restore_moved_class(fixed) is False
    assert fixed.DentalModelSeg is before


def test_the_workaround_does_not_import_anything_when_the_name_is_there(monkeypatch):
    """A no-op must not drag `saxi_nets_lightning` in behind it."""
    monkeypatch.setitem(sys.modules, "shapeaxi", None)

    assert pipeline._restore_moved_class(_HasTheClass()) is False


def test_a_shapeaxi_with_the_class_in_neither_module_is_unavailable(monkeypatch):
    """A release that moved it a third time must say so rather than fail later
    with an AttributeError from inside shapeaxi."""
    empty = types.ModuleType("shapeaxi.saxi_nets_lightning")
    monkeypatch.setitem(sys.modules, "shapeaxi.saxi_nets_lightning", empty)
    monkeypatch.setitem(sys.modules, "shapeaxi", types.ModuleType("shapeaxi"))

    with pytest.raises(ToolUnavailableError, match="2.0.2"):
        pipeline._restore_moved_class(_MissingTheClass())


# ---------------------------------------------------------------------------
# The error classes the server maps to status codes
# ---------------------------------------------------------------------------

def test_the_input_error_is_the_one_the_server_answers_422_with():
    """`dispatch.py` maps by exception class NAME, there being no shared
    package to define a base class in. The base class matters too: a plain
    ValueError escaping the tool is answered the same way."""
    assert ToolInputError.__name__ == "ToolInputError"
    assert issubclass(ToolInputError, ValueError)


def test_the_unavailable_error_is_the_one_the_server_answers_501_with():
    assert ToolUnavailableError.__name__ == "ToolUnavailableError"
    assert issubclass(ToolUnavailableError, RuntimeError)
    assert not issubclass(ToolUnavailableError, ValueError)
