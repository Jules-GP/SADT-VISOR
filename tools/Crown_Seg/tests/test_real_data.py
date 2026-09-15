"""The published test mesh, and the real checkpoint through the real engine.

Both are deselected by default (`-m 'not gpu and not models'`): a plain
`pytest` must not depend on a 16 MB mesh being staged or on a card being
present. Run them by hand and report the result in the PR --
see tests/data/README.md.
"""

import json
import os
from pathlib import Path

import pytest

from sadt_crownseg import pipeline, run

REAL_MODEL = os.environ.get("SADT_CROWNSEG_MODEL")
REAL_MESH = os.environ.get("SADT_CROWNSEG_MESH")

needs_mesh = pytest.mark.skipif(
    not REAL_MESH, reason="set SADT_CROWNSEG_MESH (see tests/data/README.md)"
)
needs_both = pytest.mark.skipif(
    not (REAL_MODEL and REAL_MESH),
    reason="set SADT_CROWNSEG_MODEL and SADT_CROWNSEG_MESH (see tests/data/README.md)",
)


def point_count(path):
    import vtk

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput().GetNumberOfPoints()


@pytest.mark.models
@needs_mesh
def test_the_published_test_mesh_is_the_one_the_readme_describes():
    """`T1_01_U_segmented.vtk`, 294 260 points, already carrying labels -- which
    is why the real-model test has to pass `skip_segmented=False` to make the
    network run at all."""
    assert point_count(REAL_MESH) == 294260
    assert pipeline.is_segmented(REAL_MESH) is True


@pytest.mark.models
@needs_both
def test_a_real_labelled_mesh_passes_through_with_its_geometry_intact():
    """The bypass on real data, and it needs no GPU: 294 260 points and their
    label array survive the .vtk round trip `_write_as_vtk` performs."""
    import tempfile

    with tempfile.TemporaryDirectory() as work:
        output = run(
            meshes=Path(REAL_MESH),
            model=Path(REAL_MODEL),
            output_dir=Path(work) / "out",
        )
        report = json.loads((output / "run_report.json").read_text())

        assert report["summary"]["already_segmented"] == 1
        assert report["summary"]["segmented"] == 0
        assert report["device"] is None

        produced = Path(report["segmented_meshes"][0])
        assert produced.name == Path(REAL_MESH).stem + "_Seg.vtk"
        assert point_count(produced) == 294260
        assert pipeline.is_segmented(str(produced))


@pytest.mark.gpu
@pytest.mark.models
@needs_both
def test_real_model_labels_a_real_mesh():
    """The real checkpoint through the real shapeaxi, on a real arch, with the
    bypass off so the network actually runs.

    `PredictedID` is the array to compare against a reference and the only one:
    it is the network's raw per-point prediction, while `Universal_ID` comes out
    of shapeaxi's own closing operation, which is not deterministic between two
    runs of the same code (README records the spread).
    """
    import tempfile

    with tempfile.TemporaryDirectory() as work:
        output = run(
            meshes=Path(REAL_MESH),
            model=Path(REAL_MODEL),
            output_dir=Path(work) / "out",
            skip_segmented=False,
        )
        report = json.loads((output / "run_report.json").read_text())

        assert report["summary"]["segmented"] == 1
        assert report["summary"]["failed"] == 0
        assert report["engine_available"] is True

        produced = Path(report["segmented_meshes"][0])
        assert pipeline.is_segmented(str(produced))
        assert point_count(produced) == 294260

        import vtk

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(produced))
        reader.Update()
        point_data = reader.GetOutput().GetPointData()
        names = {
            point_data.GetArrayName(index)
            for index in range(point_data.GetNumberOfArrays())
        }
        assert "PredictedID" in names
