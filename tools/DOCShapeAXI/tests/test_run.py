"""End-to-end tests for DOCShapeAXI.

The network and the GradCAM pass are stubbed, so everything around them runs
for real: discovery, the checkpoint catalog, the report, the suffixing, and the
regression-versus-classification split. The network itself cannot run in CI --
it needs a CUDA device and 72 MB of weights per checkpoint -- which is what the
`gpu`/`models` markers are for.
"""

import json
import os

import pytest

from sadt_docshapeaxi import _rename_with_suffix, run
from sadt_docshapeaxi import catalog, engine, pipeline


def write_surface(path):
    """A one-triangle `.vtk`, which is all discovery and naming need."""
    vtk = pytest.importorskip("vtk")

    points = vtk.vtkPoints()
    for coordinates in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
        points.InsertNextPoint(*coordinates)
    polys = vtk.vtkCellArray()
    polys.InsertNextCell(3)
    for point_id in range(3):
        polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the three things that need a GPU and a checkpoint."""
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested: "cpu")
    monkeypatch.setattr(pipeline, "load_network", lambda *a, **k: object())

    def predict(model, analysis, surfaces, mount_point, device):
        return [float(index % max(analysis.classes, 1)) for index in range(len(surfaces))]

    def explain(model, analysis, surfaces, mount_point, device, output_dir):
        written = []
        for path in surfaces:
            destination = os.path.join(output_dir, os.path.basename(path))
            write_surface(destination)
            written.append(destination)
        return written

    monkeypatch.setattr(engine, "predict", predict)
    monkeypatch.setattr(engine, "explain", explain)


def test_the_checkpoint_name_carries_the_whole_analysis():
    analysis = catalog.analysis_for("airways_4_regress.ckpt")
    assert analysis.is_regression
    assert analysis.classes == 1
    assert catalog.analysis_for("airways_2_class.ckpt").classes == 2


def test_an_unknown_checkpoint_names_the_ones_that_exist():
    with pytest.raises(ValueError) as raised:
        catalog.analysis_for("clefts_4_class.ckpt")
    # `clefts_4_class` is upstream's own spelling, and there is no such file.
    assert "cleft_4_class.ckpt" in str(raised.value)


def test_discovery_is_recursive_and_ignores_everything_but_vtk(tmp_path):
    write_surface(tmp_path / "a.vtk")
    write_surface(tmp_path / "nested" / "b.vtk")
    (tmp_path / "notes.txt").write_text("not a surface")
    (tmp_path / "scan.nii.gz").write_bytes(b"\x1f\x8b")

    found = pipeline.discover_surfaces(str(tmp_path))
    assert [os.path.basename(path) for path in found] == ["a.vtk", "b.vtk"]


def test_a_folder_with_no_surface_is_refused_before_anything_is_loaded(tmp_path):
    (tmp_path / "scan.nii.gz").write_bytes(b"\x1f\x8b")
    with pytest.raises(ValueError, match="No .vtk surface"):
        run(surfaces=str(tmp_path), model="airways_2_class.ckpt",
            output_dir=str(tmp_path / "out"))


def test_a_missing_checkpoint_is_reported_by_name(tmp_path):
    write_surface(tmp_path / "a.vtk")
    with pytest.raises(FileNotFoundError, match="airways_2_class.ckpt"):
        run(surfaces=str(tmp_path), model=str(tmp_path / "airways_2_class.ckpt"),
            output_dir=str(tmp_path / "out"))


def test_a_classification_run_reports_a_class_per_surface(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    write_surface(tmp_path / "in" / "b.vtk")
    checkpoint = tmp_path / "condyles_4_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"))

    assert [entry["class"] for entry in result["predictions"]] == [0, 1]
    assert all("score" not in entry for entry in result["predictions"])


def test_a_regression_run_reports_a_value_not_class_zero(tmp_path, monkeypatch, stubbed):
    """Upstream took an argmax for every checkpoint. A regression network has
    one output column, so its argmax is 0 for every subject -- every patient
    graded identically, with no error anywhere."""
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_4_regress.ckpt"
    checkpoint.write_bytes(b"")
    monkeypatch.setattr(engine, "predict", lambda *a, **k: [2.75])

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"))

    assert result["predictions"] == [{"surface": "a.vtk", "score": 2.75}]


def test_every_surface_is_written_once_under_its_own_name(tmp_path, stubbed):
    """Upstream wrote inside the per-class loop, to a path that did not depend
    on the class, so a four-class run wrote the same file four times."""
    write_surface(tmp_path / "in" / "a.vtk")
    write_surface(tmp_path / "in" / "b.vtk")
    checkpoint = tmp_path / "cleft_4_class.ckpt"
    checkpoint.write_bytes(b"")

    run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
        output_dir=str(tmp_path / "out"), output_suffix="_pred")

    written = sorted(os.listdir(tmp_path / "out"))
    assert written == ["DOCShapeAXI_report.json", "a_pred.vtk", "b_pred.vtk"]


def test_the_report_says_which_checkpoint_answered(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
                 output_dir=str(tmp_path / "out"), explain=False)

    report = json.loads(open(result["report"], encoding="utf-8").read())
    assert report["model"] == "airways_2_class.ckpt"
    assert report["task"] == "binary"
    assert report["classes"] == 2
    assert report["explained"] is False
    assert report["surfaces"] == 1


def test_without_explain_no_surface_is_written(tmp_path, stubbed):
    write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    run(surfaces=str(tmp_path / "in"), model=str(checkpoint),
        output_dir=str(tmp_path / "out"), explain=False)

    assert os.listdir(tmp_path / "out") == ["DOCShapeAXI_report.json"]


def test_a_single_file_is_accepted_as_well_as_a_folder(tmp_path, stubbed):
    surface = write_surface(tmp_path / "in" / "a.vtk")
    checkpoint = tmp_path / "airways_2_class.ckpt"
    checkpoint.write_bytes(b"")

    result = run(surfaces=surface, model=str(checkpoint),
                 output_dir=str(tmp_path / "out"))

    assert len(result["predictions"]) == 1


def test_an_empty_suffix_leaves_the_written_names_alone(tmp_path):
    written = [write_surface(tmp_path / "a.vtk")]
    assert _rename_with_suffix(written, "") == written


def test_a_flat_attribution_map_normalises_to_zeros_not_nan():
    """Upstream divided by `max - min` unconditionally. A network that
    attributes nothing produced a surface array of NaN, which VTK writes and
    Slicer colours as if it meant something."""
    numpy = pytest.importorskip("numpy")

    scaled = pipeline.scale_attribution(numpy.zeros((1, 8, 8), dtype="float32"))
    assert numpy.isfinite(scaled).all()
    assert not scaled.any()


def test_an_attribution_map_is_rescaled_into_minus_one_to_one():
    numpy = pytest.importorskip("numpy")

    ramp = numpy.linspace(-40, 90, 64, dtype="float32").reshape(1, 8, 8)
    scaled = pipeline.scale_attribution(ramp)
    assert scaled.shape == (1, 224, 224)
    assert -1.0001 <= scaled.min() and scaled.max() <= 1.0001


@pytest.mark.gpu
@pytest.mark.models
def test_real_model_classifies_a_real_surface(tmp_path):
    """Run by hand against the staged checkpoints; the result goes in the PR."""
    models = os.environ.get("DOCSHAPEAXI_MODELS")
    surfaces = os.environ.get("DOCSHAPEAXI_SURFACES")
    if not models or not surfaces:
        pytest.skip("set DOCSHAPEAXI_MODELS and DOCSHAPEAXI_SURFACES")

    result = run(surfaces=surfaces,
                 model=os.path.join(models, "condyles_4_class.ckpt"),
                 output_dir=str(tmp_path))
    assert result["predictions"]
