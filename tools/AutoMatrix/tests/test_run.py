"""AutoMatrix on synthetic volumes: SimpleITK does the work, for real."""

import json
import os
import sys

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import sadt_automatrix
from sadt_automatrix import pipeline


def _volume(path, value=7):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((6, 6, 6), np.int16)
    array[2:4, 2:4, 2:4] = value
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))
    return path


def _transform(path, translation=(1.0, 0.0, 0.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    tfm = sitk.TranslationTransform(3, translation)
    sitk.WriteTransform(tfm, str(path))
    return path


def _landmarks(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"markups": [{"controlPoints": [
        {"label": f"L{i}", "position": list(p), "positionStatus": "defined"}
        for i, p in enumerate(points)]}]}))
    return path


# ---------------------------------------------------------------------------
# The rule that matters most
# ---------------------------------------------------------------------------

def test_a_segmentation_is_resampled_without_inventing_labels(tmp_path):
    """Nearest neighbour for a label map, linear otherwise. Interpolating
    labels linearly produces values that were never in the input -- a voxel
    half-way between label 1 and label 3 becomes label 2, which is a different
    anatomical structure."""
    array = np.zeros((8, 8, 8), np.int16)
    array[2:5, 2:6, 2:6] = 1
    array[5:8, 2:6, 2:6] = 3
    image = sitk.GetImageFromArray(array)
    # Not half a voxel: a 0.5 shift lands exactly on the boundary, where the
    # blend of 1 and 3 truncates back to an existing label and hides the
    # effect. An off-grid shift is what a real registration produces.
    shift = sitk.TranslationTransform(3, (0.3, 0.3, 0.3))

    as_labels = sitk.GetArrayFromImage(pipeline.resample(image, shift, is_segmentation=True))
    as_scan = sitk.GetArrayFromImage(pipeline.resample(image, shift, is_segmentation=False))

    assert set(np.unique(as_labels)) <= {0, 1, 3}, "nearest neighbour invented a label"
    assert 2 in set(np.unique(as_scan)), (
        "linear interpolation did not blend, so this test no longer demonstrates "
        "what the nearest-neighbour rule protects against"
    )


def test_landmarks_move_by_the_inverse(tmp_path):
    """A transform produced by a registration maps IMAGE space; a point follows
    the opposite way. Upstream applies the inverse and so does this."""
    source = _landmarks(tmp_path / "P1_lm.mrk.json", [(10.0, 0.0, 0.0)])
    tfm = sitk.TranslationTransform(3, (4.0, 0.0, 0.0))
    out = tmp_path / "out.mrk.json"

    moved = pipeline.apply_to_landmarks(str(source), tfm, str(out))

    assert moved == 1
    position = json.loads(out.read_text())["markups"][0]["controlPoints"][0]["position"]
    assert position == pytest.approx([6.0, 0.0, 0.0]), "point moved forwards, not by the inverse"


def test_a_point_that_is_not_defined_is_left_alone(tmp_path):
    source = tmp_path / "P1_lm.mrk.json"
    source.write_text(json.dumps({"markups": [{"controlPoints": [
        {"label": "A", "position": [1.0, 2.0, 3.0], "positionStatus": "undefined"},
        {"label": "B", "position": [10.0, 0.0, 0.0], "positionStatus": "defined"},
    ]}]}))
    out = tmp_path / "out.mrk.json"

    assert pipeline.apply_to_landmarks(str(source), sitk.TranslationTransform(3, (4.0, 0, 0)), str(out)) == 1
    points = json.loads(out.read_text())["markups"][0]["controlPoints"]
    assert points[0]["position"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Discovery and pairing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,image,landmark", [
    ("P1_T1.nii.gz", True, False),
    ("P1_T1.nrrd", True, False),
    ("P1_lm_Pred.mrk.json", False, True),
    # A landmark file ends in .json but must never be read as an image.
    ("P1_something.json", False, False),
    ("notes.txt", False, False),
])
def test_what_counts_as_transformable(name, image, landmark):
    assert pipeline.is_image_file(name) is image
    assert pipeline.is_landmark_file(name) is landmark


def test_a_patient_with_no_transform_is_named_not_dropped(tmp_path):
    """Upstream skipped such a file in silence, so a run could transform 3 of
    40 and look complete."""
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    _volume(tmp_path / "in" / "P2_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    report = json.loads((out / "AutoMatrix_report.json").read_text())
    assert report["without_a_transform"] == ["P2"]
    assert list(report["patients"]) == ["P1"]


def test_a_transform_with_no_file_is_named_too(tmp_path):
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_transform.tfm")
    _transform(tmp_path / "tfm" / "P9_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    report = json.loads((out / "AutoMatrix_report.json").read_text())
    assert report["transforms_without_a_file"] == ["P9"]


def test_several_transforms_of_one_patient_are_all_applied(tmp_path):
    """And they have to be told apart, which is what the name flag is for."""
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_CBReg_matrix.tfm", (1.0, 0, 0))
    _transform(tmp_path / "tfm" / "P1_MANDReg_matrix.tfm", (0, 2.0, 0))

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out",
                              name_output_after_transform=True)

    written = sorted(p.name for p in out.rglob("*.nii.gz"))
    assert len(written) == 2, f"expected one output per transform, got {written}"
    assert len(set(written)) == 2, "two transforms wrote the same file name"


def test_nothing_transformable_is_refused(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "notes.txt").write_text("nothing")
    _transform(tmp_path / "tfm" / "P1_transform.tfm")

    with pytest.raises(ValueError, match="No scan, segmentation or landmark"):
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")


def test_no_transform_at_all_is_refused(tmp_path):
    _volume(tmp_path / "in" / "P1_T1.nii.gz")
    (tmp_path / "tfm").mkdir()

    with pytest.raises(ValueError, match="No transform found"):
        sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                            output_dir=tmp_path / "out")


def test_the_input_tree_is_mirrored(tmp_path):
    _volume(tmp_path / "in" / "siteA" / "P1_T1.nii.gz")
    _transform(tmp_path / "tfm" / "P1_transform.tfm")

    out = sadt_automatrix.run(files=tmp_path / "in", transforms=tmp_path / "tfm",
                              output_dir=tmp_path / "out")

    assert list((out / "siteA").glob("*.nii.gz")), "the subfolder was flattened"
