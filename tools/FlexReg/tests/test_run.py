"""FlexReg's unit tests: no GPU, no weights, no network.

The patch itself needs a CUDA device (upstream's propagation calls `.cuda()`
unconditionally), so building one is marked `gpu` and skipped in CI. Everything
around it -- reading, the coordinate convention, discovery, the argument rules,
the report -- runs for real against synthetic surfaces.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

import sadt_flexreg
from sadt_flexreg import pipeline
from sadt_flexreg.pipeline import ToolInputError


def _write_surface(path, labelled=True, points=None):
    """A minimal polydata, optionally carrying a per-point label array."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coordinates = points if points is not None else (
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1),
    )
    vtk_points = vtk.vtkPoints()
    for coordinate in coordinates:
        vtk_points.InsertNextPoint(*coordinate)

    polys = vtk.vtkCellArray()
    for triangle in ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)):
        polys.InsertNextCell(3)
        for point_id in triangle:
            polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(vtk_points)
    surface.SetPolys(polys)

    if labelled:
        labels = numpy_to_vtk(np.arange(len(coordinates), dtype=np.int32), deep=True)
        labels.SetName("Universal_ID")
        surface.GetPointData().AddArray(labels)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(surface)
    writer.Write()
    return str(path)


# ---------------------------------------------------------------------------
# The coordinate convention


def test_a_surface_leaves_in_the_convention_it_arrived_in(tmp_path):
    """Upstream flipped LPS to RAS on read and never flipped back, so the file
    it wrote was RAS. Invisible while both ends are Slicer, wrong the moment
    anything else reads the result."""
    source = _write_surface(tmp_path / "in.vtk", points=((3.0, 5.0, 7.0),) * 4)

    surface = pipeline.read_surface(source)
    written = pipeline.write_surface(surface, str(tmp_path / "out.vtk"))

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(written)
    reader.Update()
    assert reader.GetOutput().GetPoint(0) == pytest.approx((3.0, 5.0, 7.0))


def test_reading_converts_to_ras(tmp_path):
    """The engines work in RAS, which is the flip's whole purpose."""
    source = _write_surface(tmp_path / "in.vtk", points=((3.0, 5.0, 7.0),) * 4)

    surface = pipeline.read_surface(source)

    assert surface.GetPoint(0) == pytest.approx((-3.0, -5.0, 7.0))


# ---------------------------------------------------------------------------
# What it reads


def test_a_stl_is_read_rather_than_silently_empty(tmp_path):
    """Upstream used vtkPolyDataReader for everything, so a .stl came back as an
    empty mesh and failed much later without naming the file."""
    surface = vtk.vtkSphereSource()
    surface.Update()
    path = tmp_path / "arch.stl"
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface.GetOutput())
    writer.Write()

    assert pipeline.read_surface(str(path)).GetNumberOfPoints() > 0


def test_a_file_that_is_not_a_surface_is_refused_by_name(tmp_path):
    path = tmp_path / "scan.nii.gz"
    path.write_bytes(b"not a surface")

    with pytest.raises(ToolInputError, match="is not a surface"):
        pipeline.read_surface(str(path))


def test_a_folder_is_walked_and_sorted(tmp_path):
    """A cohort is one call: the server pays a process start-up per call, and
    sorted because readdir order varies between filesystems."""
    _write_surface(tmp_path / "b" / "second.vtk")
    _write_surface(tmp_path / "a" / "first.vtk")

    found = pipeline.surfaces_in(str(tmp_path))

    assert [os.path.basename(path) for path in found] == ["first.vtk", "second.vtk"]


def test_a_folder_with_no_surface_says_what_it_wanted(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing here")

    with pytest.raises(ToolInputError, match=r"\.vtk"):
        pipeline.surfaces_in(str(tmp_path))


# ---------------------------------------------------------------------------
# The argument rules


def test_registering_without_a_reference_is_refused_before_anything_is_read(tmp_path):
    """Reading a cohort first only delays the same answer."""
    with pytest.raises(ToolInputError, match="reference"):
        sadt_flexreg.run(
            surfaces=tmp_path / "absent",
            output_dir=tmp_path / "out",
            mode="Register",
        )


def test_registering_on_a_patch_that_is_not_there_says_so(tmp_path):
    """A mesh with no patch array has nothing to register on, and an ICP run on
    an empty selection is the failure this replaces."""
    plain = pipeline.read_surface(_write_surface(tmp_path / "plain.vtk"))

    with pytest.raises(ToolInputError, match="Butterfly"):
        pipeline.register(plain, plain, pipeline.BUTTERFLY_ARRAY)


# ---------------------------------------------------------------------------
# The transform


def test_the_written_transform_maps_registered_back_onto_the_original(tmp_path):
    """Direction asserted rather than assumed: a Slicer transform node maps the
    space, not the points, so getting it backwards is silent."""
    import SimpleITK as sitk

    matrix = np.eye(4)
    matrix[:3, 3] = (10.0, 0.0, 0.0)
    path = pipeline.write_transform(matrix, str(tmp_path / "t.tfm"))

    transform = sitk.ReadTransform(path)
    parameters = np.asarray(transform.GetParameters())
    # LPS: the flip negates x, and the transform is the inverse of the applied
    # one, so a +10 mm move in RAS is a +10 mm translation here.
    assert parameters[9] == pytest.approx(10.0)


def test_an_identity_registration_writes_an_identity_transform(tmp_path):
    import SimpleITK as sitk

    path = pipeline.write_transform(np.eye(4), str(tmp_path / "t.tfm"))

    parameters = np.asarray(sitk.ReadTransform(path).GetParameters())
    assert parameters[:9] == pytest.approx(np.eye(3).flatten())
    assert parameters[9:] == pytest.approx((0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Merging patches


def test_several_drawn_patches_merge_into_the_one_the_registration_reads(tmp_path):
    """A mesh can carry Butterfly1..N; the registration reads `Butterfly`."""
    surface = pipeline.read_surface(_write_surface(tmp_path / "in.vtk"))
    for index, values in ((1, [1, 0, 0, 0]), (2, [0, 0, 1, 0])):
        array = numpy_to_vtk(np.array(values, dtype=np.int32), deep=True)
        array.SetName("Butterfly{}".format(index))
        surface.GetPointData().AddArray(array)

    pipeline.merge_patches(surface)

    from vtk.util.numpy_support import vtk_to_numpy
    merged = vtk_to_numpy(surface.GetPointData().GetArray(pipeline.BUTTERFLY_ARRAY))
    assert merged.tolist() == [1, 0, 1, 0]


def test_a_mesh_with_no_patch_is_left_alone(tmp_path):
    surface = pipeline.read_surface(_write_surface(tmp_path / "in.vtk"))

    pipeline.merge_patches(surface)

    assert not surface.GetPointData().HasArray(pipeline.BUTTERFLY_ARRAY)


# ---------------------------------------------------------------------------
# ApplyTransform, whose branches were undefined names


def test_applying_a_transform_to_something_that_is_not_a_surface_is_named(tmp_path):
    """Upstream dispatched on type and called `TransformDict`/`TransformList`,
    neither of which exists in this module: the file was copied from AREG_IOS
    without them, so both branches raised NameError on a path only a real run
    reaches. Refused by name now, rather than returning the input unchanged --
    which would be a registration that quietly did nothing."""
    from sadt_flexreg.transformation import ApplyTransform

    with pytest.raises(TypeError, match="vtkPolyData"):
        ApplyTransform({"a": 1}, np.eye(4))


# ---------------------------------------------------------------------------
# The batch


def test_one_unusable_surface_does_not_cost_the_others(tmp_path):
    """Reported per surface and the batch continues: one patient a caller has to
    fix must not lose the other thirty-nine."""
    _write_surface(tmp_path / "good.vtk")
    (tmp_path / "bad.vtk").write_bytes(b"truncated")

    sadt_flexreg.run(surfaces=tmp_path, output_dir=tmp_path / "out", mode="Patch",
                     patch="Mucogingival line")

    report = json.loads((tmp_path / "out" / "FlexReg_report.json").read_text())
    assert report["summary"]["total"] == 2
    assert report["summary"]["failed"] == 1
    assert report["summary"]["processed"] == 1


def test_a_batch_where_nothing_worked_raises_rather_than_reporting_success(tmp_path):
    (tmp_path / "bad.vtk").write_bytes(b"truncated")

    with pytest.raises(ToolInputError, match="None of the"):
        sadt_flexreg.run(surfaces=tmp_path, output_dir=tmp_path / "out", mode="Patch")


def test_the_output_mirrors_the_input_tree(tmp_path):
    """Keyed by path relative to the input root, so two patients named the same
    in different folders do not overwrite each other."""
    _write_surface(tmp_path / "p1" / "arch.vtk")
    _write_surface(tmp_path / "p2" / "arch.vtk")

    sadt_flexreg.run(surfaces=tmp_path, output_dir=tmp_path / "out", mode="Patch",
                     patch="Mucogingival line")

    assert (tmp_path / "out" / "p1" / "arch_Reg.vtk").is_file()
    assert (tmp_path / "out" / "p2" / "arch_Reg.vtk").is_file()


# ---------------------------------------------------------------------------
# The corner pairs


def test_a_corner_is_one_position_rather_than_two_settings(tmp_path):
    """Each pair reaches the engine as (ratio, adjust), in that order.

    Upstream had ten flat floats read off a Qt form. They are five pairs here so
    the panel gives each corner a 2D pad whose knob sits where the point sits on
    the arch; getting the order backwards would move a corner sideways when the
    clinician pushed it forward, which nothing downstream could detect.
    """
    seen = {}

    def _capture(surface, teeth, ratios, adjustments, index, shift_lr, shift_ap):
        seen.update(ratios=dict(ratios), adjustments=dict(adjustments),
                    shift=(shift_lr, shift_ap))
        return surface

    import sadt_flexreg as tool
    original = tool.build_butterfly
    tool.build_butterfly = _capture
    try:
        _write_surface(tmp_path / "arch.vtk")
        tool.run(
            surfaces=tmp_path / "arch.vtk",
            output_dir=tmp_path / "out",
            mode="Patch",
            anterior_right=(0.8, -2.0),
            shift=(3.0, -4.0),
        )
    finally:
        tool.build_butterfly = original

    assert seen["ratios"]["anterior_right"] == 0.8
    assert seen["adjustments"]["anterior_right"] == -2.0
    assert seen["shift"] == (3.0, -4.0)


def test_every_corner_keeps_its_own_pair(tmp_path):
    """Four pads, four corners: a transposition here would be invisible in a
    report and visible only as a misplaced patch."""
    seen = {}

    def _capture(surface, teeth, ratios, adjustments, index, shift_lr, shift_ap):
        seen.update(ratios=dict(ratios), adjustments=dict(adjustments))
        return surface

    import sadt_flexreg as tool
    original = tool.build_butterfly
    tool.build_butterfly = _capture
    try:
        _write_surface(tmp_path / "arch.vtk")
        tool.run(
            surfaces=tmp_path / "arch.vtk", output_dir=tmp_path / "out", mode="Patch",
            anterior_right=(0.1, 1.0), anterior_left=(0.2, 2.0),
            posterior_right=(0.3, 3.0), posterior_left=(0.4, 4.0),
        )
    finally:
        tool.build_butterfly = original

    assert seen["ratios"] == {
        "anterior_right": 0.1, "anterior_left": 0.2,
        "posterior_right": 0.3, "posterior_left": 0.4,
    }
    assert seen["adjustments"] == {
        "anterior_right": 1.0, "anterior_left": 2.0,
        "posterior_right": 3.0, "posterior_left": 4.0,
    }


def test_a_tooth_the_arch_does_not_have_fails_the_surface(tmp_path, monkeypatch):
    """Upstream logged `ToothNoExist` and returned.

    A lower arch has none of the palate teeth, so every run on one wrote the
    mesh back with no Butterfly array and reported "1 processed, 0 failed" --
    a success with nothing in it. Found by running the lower arch test file.
    """
    from sadt_flexreg import make_butterfly
    from sadt_flexreg.util import ToothNoExist

    def _missing(**_kwargs):
        raise ToothNoExist("UR6")

    monkeypatch.setattr(make_butterfly, "butterflyPatch", _missing)
    _write_surface(tmp_path / "lower.vtk")

    with pytest.raises(ToolInputError, match="None of the"):
        sadt_flexreg.run(surfaces=tmp_path / "lower.vtk",
                         output_dir=tmp_path / "out", mode="Patch")

    report = json.loads((tmp_path / "out" / "FlexReg_report.json").read_text())
    assert report["summary"] == {"total": 1, "processed": 0, "failed": 1}


def test_a_run_where_nothing_worked_still_writes_its_report(tmp_path):
    """The refusal points at the report, so the report has to exist. It was the
    one case that wrote none."""
    (tmp_path / "bad.vtk").write_bytes(b"truncated")

    with pytest.raises(ToolInputError, match="FlexReg_report.json"):
        sadt_flexreg.run(surfaces=tmp_path, output_dir=tmp_path / "out", mode="Patch")

    assert (tmp_path / "out" / "FlexReg_report.json").is_file()
