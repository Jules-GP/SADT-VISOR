"""End-to-end test for the template tool.

A tool is not done until a test has actually run it: an import check proves
nothing about a segmentation pipeline. Every `tools/<name>/tests/test_run.py`
must call `run()` on real input and assert on the files it produced -- they
exist, they are a plausible size, and where a reference output is available,
they match it within a tolerance the README documents.

Fixtures are built here rather than committed. Test data goes in `tests/data/`
only as a download script plus checksums; no large binaries and no patient
data in git, ever.
"""

import numpy as np
import pytest

from sadt_template import progress, run
from sadt_template.pipeline import ToolInputError


@pytest.fixture
def scans(tmp_path):
    """A folder of two arrays with known statistics."""
    folder = tmp_path / "scans"
    folder.mkdir()
    np.save(folder / "a.npy", np.arange(100, dtype=np.float32))
    np.save(folder / "b.npy", np.full((10, 10), 7.0, dtype=np.float32))
    return folder


def test_run_on_a_folder(scans, tmp_path):
    output = run(scans=scans, output_dir=tmp_path / "out")

    assert output.exists() and output.stat().st_size > 0
    lines = output.read_text().splitlines()
    assert len(lines) == 2, "one row per scan, in sorted order"
    assert lines[0].startswith("scan=a.npy")
    # arange(100) above threshold 0 is 1..99: mean 50, max 99.
    assert "mean=50.0" in lines[0] and "max=99.0" in lines[0]


def test_run_on_a_single_file(scans, tmp_path):
    """The same argument takes one file -- batch-capable, not batch-only."""
    output = run(scans=scans / "b.npy", output_dir=tmp_path / "out")

    assert output.read_text().strip() == "scan=b.npy voxels=100 mean=7.0 max=7.0"


def test_writes_nothing_outside_the_output_directory(scans, tmp_path):
    before = sorted(scans.iterdir())
    run(scans=scans, output_dir=tmp_path / "out", per_scan_report=True)

    assert sorted(scans.iterdir()) == before
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        "summary.txt",
        "a.txt",
        "b.txt",
    }


def test_optional_arguments_are_honoured(scans, tmp_path):
    output = run(
        scans=scans,
        output_dir=tmp_path / "out",
        metrics=["min", "std"],
        threshold=50.0,
    )

    row = output.read_text().splitlines()[0]
    assert "min=51.0" in row and "std=" in row and "mean=" not in row


def test_bad_input_is_reported_as_such(tmp_path):
    """Bad input raises ToolInputError; a bug raises anything else."""
    with pytest.raises(ToolInputError, match="does not exist"):
        run(scans=tmp_path / "missing", output_dir=tmp_path / "out")

    with pytest.raises(ToolInputError, match="Unknown metric"):
        run(scans=tmp_path, output_dir=tmp_path / "out", metrics=["median"])


@pytest.mark.gpu
def test_runs_on_gpu():
    """Marked tests are skipped in CI and run by hand before opening the PR.

    Delete this in a tool with no GPU path; keep it, and state in the PR that
    you ran `pytest -m gpu` and what came out.
    """
    pytest.skip("the template has no GPU path")


def test_pooled_reduction_collapses_the_batch_to_one_row(scans, tmp_path):
    """The `Literal` single-select: exactly one of a fixed set."""
    output = run(scans=scans, output_dir=tmp_path / "out", reduction="pooled")

    lines = output.read_text().splitlines()
    assert len(lines) == 1
    # arange(100) above 0 is 1..99, plus 100 voxels of 7.0.
    assert lines[0].startswith("scan=all voxels=199")


def test_an_option_outside_the_published_set_is_refused(scans, tmp_path):
    """`Literal` is published as `choices`, not enforced by Python.

    The runner calls run(**params) from a JSON object, so a stale client or a
    direct caller can still send anything. The tool checks.
    """
    with pytest.raises(ToolInputError, match="Unknown reduction"):
        run(scans=scans, output_dir=tmp_path / "out", reduction="median")


# ---------------------------------------------------------------------------
# Progress -- the channel a cohort loop reports on
# ---------------------------------------------------------------------------

def _events(path):
    """Every event written so far, parsed. One JSON object per line."""
    import json

    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_progress_is_silent_when_no_server_asked_for_it(monkeypatch, capsys):
    """The variable is unset in a checkout and in every other test in this file.

    Nothing may be written, nothing printed, and nothing raised -- which is
    what lets a tool call `report` unconditionally instead of guarding it.
    """
    monkeypatch.delenv(progress.VARIABLE, raising=False)

    progress.report(1, 10, "scan")
    progress.emit(None, "anything")

    assert capsys.readouterr() == ("", "")


def test_one_well_formed_line_per_event(tmp_path, monkeypatch):
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(events_file))

    progress.report(14, 40, "scan")
    progress.emit(None, "an opaque phase")

    events = _events(events_file)
    assert events[0] == {"fraction": 0.325, "message": "scan 14 of 40"}
    # None, not a number: the honest answer where the tool cannot see inside.
    assert events[1] == {"fraction": None, "message": "an opaque phase"}


def test_a_message_is_truncated_to_what_the_server_keeps(tmp_path, monkeypatch):
    """Under PIPE_BUF, below which POSIX makes an O_APPEND write atomic -- which
    is what stops a supervised chain's events from interleaving."""
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(events_file))

    progress.emit(0.5, "x" * 5000)

    line = events_file.read_bytes()
    assert len(line) < progress.PIPE_BUF
    assert _events(events_file)[0]["message"] == "x" * progress.MAX_MESSAGE


def test_a_failure_to_report_never_reaches_the_tool(tmp_path, monkeypatch):
    """Telemetry must not be able to fail a run that is otherwise fine."""
    monkeypatch.setenv(progress.VARIABLE, str(tmp_path / "no" / "such" / "dir" / "e"))
    progress.report(1, 2, "scan")  # a directory that does not exist

    monkeypatch.setenv(progress.VARIABLE, str(tmp_path / "events.jsonl"))
    progress.emit("not a number", "nor is the fraction checked by the caller")

    assert not (tmp_path / "events.jsonl").exists()


def test_a_cohort_loop_reports_one_event_per_scan(scans, tmp_path, monkeypatch):
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(events_file))

    run(scans=scans, output_dir=tmp_path / "out")

    events = _events(events_file)
    assert [event["message"] for event in events] == ["scan 1 of 2", "scan 2 of 2"]
    # Rising, and starting at zero: the fraction is the share of the batch
    # already behind the item, so it never counts one that is still running.
    fractions = [event["fraction"] for event in events]
    assert fractions == sorted(fractions) and fractions[0] == 0.0
