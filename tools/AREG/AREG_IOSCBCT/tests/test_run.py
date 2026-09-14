"""AREG_IOSCBCT's unit tests: no GPU, no weights, no network.

Split out of the single `tools/AREG/tests/test_run.py` AREG had before it
became three tools; see AREG_CBCT/tests/test_run.py for why none of them ran.

The tools this one drives are stood in for by a fake supervisor, which is all a
tool can see of them: five members, duck-typed, nothing imported across venvs.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from sadt_areg_ioscbct import dispatch, pipeline, run, tools
from sadt_areg_common import catalogs, pairing
from sadt_areg_common.errors import SupervisorRequired, ToolInputError


class FakeSup:
    """A supervisor, as a tool sees one. Records what it was asked for.

    `outputs` maps a tool name to a callable taking the parameters it was sent
    and returning the directory it "produced", so a test can plant results
    without any of the real tools existing.
    """

    def __init__(self, tmp_path, outputs=None):
        self.out = Path(tmp_path) / "out"
        self.tmp = Path(tmp_path) / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.outputs = outputs or {}
        self.calls = []
        self.messages = []

    def run(self, tool, **params):
        self.calls.append((tool, params))
        maker = self.outputs.get(tool)
        if maker is None:
            raise AssertionError(f"nothing planted for {tool!r} in this test")
        return Path(maker(params))

    def progress(self, fraction, message):
        self.messages.append((fraction, message))

    def log(self, message):
        self.messages.append((None, message))


def _phantom(size=48, seed=0, spacing=0.8, origin=(-140.0, -90.0, 60.0)):
    """A textured volume with an origin far from zero.

    Far from zero on purpose: that is the condition under which elastix's
    centre of rotation matters, and a phantom centred on the origin would let
    the bug this suite pins pass unnoticed.
    """
    rng = np.random.default_rng(seed)
    volume = rng.random((size,) * 3).astype(np.float32) * 120
    zz, yy, xx = np.meshgrid(*[np.arange(size)] * 3, indexing="ij")
    half = size // 2
    volume += 1400 * (
        ((zz - half) ** 2 / 180 + (yy - half + 2) ** 2 / 140 + (xx - half - 2) ** 2 / 160) < 1
    )
    volume += 900 * (
        ((zz - half + 12) ** 2 / 40 + (yy - half - 10) ** 2 / 35 + (xx - half + 12) ** 2 / 30) < 1
    )
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((spacing,) * 3)
    image.SetOrigin(origin)
    return image


def _moved(image, rotation=(0.04, -0.025, 0.03), translation=(1.2, -1.6, 0.9)):
    """`image` displaced by a known rigid transform, and that transform."""
    truth = sitk.Euler3DTransform()
    size = np.array(image.GetSize()) / 2.0
    truth.SetCenter(image.TransformContinuousIndexToPhysicalPoint(size.tolist()))
    truth.SetRotation(*rotation)
    truth.SetTranslation(translation)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(truth.GetInverse())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image), truth


def _write(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(image, path, useCompression=True)
    return path


def _full_mask(image):
    mask = sitk.GetImageFromArray(np.ones(sitk.GetArrayViewFromImage(image).shape, np.uint8))
    mask.CopyInformation(image)
    return mask


def _grid_mesh(rows=12, columns=12, spacing=1.0):
    """A flat triangulated grid, the smallest thing with a real adjacency."""
    points = vtk.vtkPoints()
    for row in range(rows):
        for column in range(columns):
            points.InsertNextPoint(column * spacing, row * spacing, 0.0)

    triangles = vtk.vtkCellArray()
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            for corners in ((a, a + 1, a + columns), (a + 1, a + columns + 1, a + columns)):
                triangle = vtk.vtkTriangle()
                for index, corner in enumerate(corners):
                    triangle.GetPointIds().SetId(index, corner)
                triangles.InsertNextCell(triangle)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(triangles)
    return mesh


def test_every_tool_is_named_by_string():
    """`sup.run("ASO", ...)`, never `sup.ASO(...)`. A typo in a string is
    greppable and tools.py is the whole call graph; a typo in an attribute is an
    AttributeError an hour into a job.

    Here rather than in AREG_CBCT, which was where the single pre-split suite
    left it: this is the tool that drives all four, so it is the only one whose
    tools.py can be expected to name all four.
    """
    source = open(tools.__file__, encoding="utf-8").read()
    assert 'sup.run("' in source
    for tool in ("Crown_Seg", "ALI_CBCT", "ALI_IOS", "ASO"):
        assert f'"{tool}"' in source, tool


# ---------------------------------------------------------------------------
# Progress -- the waypoints, and the loop after them
# ---------------------------------------------------------------------------

def _events(path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def test_each_step_says_which_tool_the_run_is_inside(tmp_path):
    """The three waypoints, which `FakeSup.messages` has recorded and nothing
    read until now.

    They are what a watcher has instead of a frozen bar: a fully-automated run
    spends most of its time inside ALI and ASO, and without them the panel
    cannot say which. They rise, and none of them names a file.
    """
    planted = tmp_path / "planted"
    planted.mkdir()
    sup = FakeSup(tmp_path, {name: (lambda params: planted) for name in
                             ("ALI_CBCT", "ALI_IOS", "ASO")})

    tools.predict_cbct_landmarks(sup, str(tmp_path), "")
    tools.predict_ios_landmarks(sup, str(tmp_path), "")
    tools.orient_cbct(sup, str(tmp_path), str(tmp_path), "")

    fractions = [fraction for fraction, _message in sup.messages]
    assert fractions == [0.1, 0.3, 0.5]
    assert [message for _fraction, message in sup.messages] == [
        "predicting CBCT landmarks with ALI_CBCT",
        "predicting intraoral landmarks with ALI_IOS",
        "orienting the CBCT with ASO",
    ]


def test_the_registration_loop_starts_where_the_waypoints_stopped(tmp_path, monkeypatch):
    """A mode that predicted nothing must not report itself as 60% done.

    The waypoints above only fire in the two modes that call other tools; the
    Registration mode calls none, so its loop owns the whole bar. Getting this
    wrong is invisible in a result and obvious to whoever is watching.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    for patient in ("P1", "P2"):
        _write(_phantom(size=8), str(tmp_path / "cbct" / f"{patient}_scan.nii.gz"))
        (tmp_path / "ios").mkdir(exist_ok=True)
        (tmp_path / "ios" / f"{patient}_Upper.vtk").write_text("")

    # Everything inside the loop is stood in for; the loop itself, which is
    # what reports, runs for real.
    monkeypatch.setattr(dispatch, "_landmarks_by_jaw", lambda root: {"any": {"A": [0.0, 0.0, 0.0]}})
    monkeypatch.setattr(dispatch, "_surface_points", lambda path: (None, np.zeros((3, 3))))
    monkeypatch.setattr(
        dispatch.pipeline, "register_one",
        lambda points, moving, fixed, max_dist: (np.eye(4), {"rms": 0.0}),
    )
    monkeypatch.setattr(
        dispatch, "_write_surface",
        lambda surface, points, path: (
            os.makedirs(os.path.dirname(path), exist_ok=True), open(path, "w").close()
        ),
    )

    for start, expected in ((0.0, [0.0, 0.5]), (0.6, [0.6, 0.8])):
        events_file.write_text("")
        report = {"patients": {}}
        dispatch.register(
            ios_dir=str(tmp_path / "ios"), cbct_dir=str(tmp_path / "cbct"),
            ios_landmark_dir=str(tmp_path), cbct_landmark_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"), suffix="Reg", report=report,
            max_dist=1.0, progress_start=start,
        )
        events = _events(events_file)
        assert [event["message"] for event in events] == ["patient 1 of 2", "patient 2 of 2"]
        assert [event["fraction"] for event in events] == expected


# ---------------------------------------------------------------------------
# What reaches the log: how many, never which
# ---------------------------------------------------------------------------

def test_the_unpaired_patients_are_counted_in_the_log_not_named(tmp_path, caplog):
    """A patient key is the caller's own file name -- and here it can BE the
    file's stem: `_patient_key` falls back to it when a name holds no digit.

    A tool's stderr is captured to a file in the job directory, and on a FAILED
    run the server copies its tail into its own persistent log, so a key
    written here outlives the run. The returned mapping still names every one
    of them, and the run report carries it back to whoever sent the data.
    """
    (tmp_path / "ios").mkdir()
    (tmp_path / "cbct").mkdir()
    (tmp_path / "ios" / "P001_Upper.vtk").write_text("")
    (tmp_path / "cbct" / "P001_scan.nii.gz").write_text("")
    (tmp_path / "ios" / "Smith_John_Upper.vtk").write_text("")

    with caplog.at_level(logging.INFO, logger="sadt_areg_ioscbct.pipeline"):
        paired, unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))

    assert list(paired) == ["1"]
    assert "Smith_John_Upper" in unpaired, "the caller is still told which"

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Not registered, only one modality present: 1 patient(s) with no CBCT, "
        "0 with no intraoral scan"
    ], messages
