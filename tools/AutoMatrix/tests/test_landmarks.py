"""Moving a Slicer markups file's points.

The INVERSE is applied. A transform produced by a registration maps IMAGE
space; a point follows the opposite way, and getting it backwards is silent --
the file is written, the count is right, and every landmark sits twice as far
from where it belongs as it started.
"""

import json

import pytest
import SimpleITK as sitk

from sadt_automatrix import pipeline


def _points(path):
    return [
        point["position"]
        for group in json.loads(path.read_text())["markups"]
        for point in group["controlPoints"]
    ]


def _rotation_about_z():
    """A quarter turn about z, plus a translation, as an affine.

    Written as a permutation matrix rather than as an Euler angle on purpose:
    cos(pi/2) is 6.1e-17 rather than 0, and the assertions below are exact.
    Forward, this maps (x, y, z) to (-y, x, z) + (1, 2, 3).
    """
    affine = sitk.AffineTransform(3)
    affine.SetMatrix([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    affine.SetTranslation([1.0, 2.0, 3.0])
    return affine


# ---------------------------------------------------------------------------
# The direction, pinned to the float
# ---------------------------------------------------------------------------

def test_a_rotation_and_a_translation_are_applied_backwards(tmp_path, landmark_file):
    """By hand: the inverse is p -> R^T (p - t). For p = (2, 0, 0),
    p - t = (1, -2, -3) and R^T (1, -2, -3) = (-2, -1, -3).

    The forward transform would put that point at (1, 4, 3), which is the
    number this test exists to refuse."""
    source = landmark_file(tmp_path / "P1_lm.mrk.json", [(2.0, 0.0, 0.0)])
    out = tmp_path / "out.mrk.json"

    assert pipeline.apply_to_landmarks(str(source), _rotation_about_z(), str(out)) == 1

    assert _points(out) == [[-2.0, -1.0, -3.0]]
    assert _points(out) != [[1.0, 4.0, 3.0]], "the forward transform was applied"


def test_the_origin_lands_where_the_inverse_puts_it(tmp_path, landmark_file):
    """R^T (0 - t) = R^T (-1, -2, -3) = (-2, 1, -3)."""
    source = landmark_file(tmp_path / "P1_lm.mrk.json", [(0.0, 0.0, 0.0)])
    out = tmp_path / "out.mrk.json"

    pipeline.apply_to_landmarks(str(source), _rotation_about_z(), str(out))

    assert _points(out) == [[-2.0, 1.0, -3.0]]


def test_a_scaling_is_divided_out_rather_than_multiplied_in(tmp_path, landmark_file):
    """diag(2, 4, 0.5) with t = (1, 1, 1), inverse of (3, 5, 2):
    ((3-1)/2, (5-1)/4, (2-1)/0.5) = (1, 1, 2). Every division is exact in
    binary, so this is an equality and not an approximation."""
    affine = sitk.AffineTransform(3)
    affine.SetMatrix([2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.5])
    affine.SetTranslation([1.0, 1.0, 1.0])
    source = landmark_file(tmp_path / "P1_lm.mrk.json", [(3.0, 5.0, 2.0)])
    out = tmp_path / "out.mrk.json"

    pipeline.apply_to_landmarks(str(source), affine, str(out))

    assert _points(out) == [[1.0, 1.0, 2.0]]


def test_a_translation_is_subtracted(tmp_path, landmark_file):
    source = landmark_file(tmp_path / "P1_lm.mrk.json", [(10.0, -3.0, 0.5)])
    out = tmp_path / "out.mrk.json"

    pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (4.0, 1.0, -0.5)), str(out))

    assert _points(out) == [[6.0, -4.0, 1.0]]


def test_applying_a_transform_then_its_inverse_gives_the_point_back(tmp_path, landmark_file):
    """The two directions compose to the identity, which is what says the
    inverse is a real inverse and not a sign flip that happens to look right on
    a translation."""
    source = landmark_file(tmp_path / "P1_lm.mrk.json", [(2.0, -7.0, 4.0)])
    once, twice = tmp_path / "once.mrk.json", tmp_path / "twice.mrk.json"
    transform = _rotation_about_z()

    pipeline.apply_to_landmarks(str(source), transform, str(once))
    pipeline.apply_to_landmarks(str(once), transform.GetInverse(), str(twice))

    assert _points(twice) == [[2.0, -7.0, 4.0]]


def test_every_point_of_every_group_moves(tmp_path):
    """A markups file may hold several nodes -- ALI writes one group, ASO reads
    several -- and a loop that only visited the first would leave half a
    patient's landmarks behind."""
    source = tmp_path / "P1_lm.mrk.json"
    source.write_text(json.dumps({"markups": [
        {"controlPoints": [
            {"label": "A", "position": [10.0, 0.0, 0.0], "positionStatus": "defined"},
            {"label": "B", "position": [0.0, 10.0, 0.0], "positionStatus": "defined"}]},
        {"controlPoints": [
            {"label": "C", "position": [0.0, 0.0, 10.0], "positionStatus": "defined"}]},
    ]}))
    out = tmp_path / "out.mrk.json"

    assert pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (1.0, 1.0, 1.0)), str(out)) == 3
    assert _points(out) == [[9.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, 9.0]]


# ---------------------------------------------------------------------------
# The points that must be left alone
# ---------------------------------------------------------------------------

def test_a_file_with_no_markups_at_all_is_written_and_moves_nothing(tmp_path):
    source = tmp_path / "P1_lm.mrk.json"
    source.write_text(json.dumps({"markups": []}))
    out = tmp_path / "out.mrk.json"

    assert pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (1.0, 0.0, 0.0)), str(out)) == 0
    assert json.loads(out.read_text()) == {"markups": []}


def test_a_group_with_no_control_points_is_not_an_error(tmp_path):
    source = tmp_path / "P1_lm.mrk.json"
    source.write_text(json.dumps({"markups": [{"type": "Fiducial"}]}))
    out = tmp_path / "out.mrk.json"

    assert pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (1.0, 0.0, 0.0)), str(out)) == 0
    assert json.loads(out.read_text())["markups"] == [{"type": "Fiducial"}]


def test_a_document_with_no_markups_key_is_not_an_error(tmp_path):
    source = tmp_path / "P1_lm.mrk.json"
    source.write_text(json.dumps({"@schema": "markups-schema-v1.0.0.json#"}))
    out = tmp_path / "out.mrk.json"

    assert pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (1.0, 0.0, 0.0)), str(out)) == 0


@pytest.mark.parametrize("position", [
    None,                       # the key is absent
    [1.0, 2.0],                 # two coordinates
    [1.0, 2.0, 3.0, 4.0],       # four
    "1.0 2.0 3.0",              # a string
    {"x": 1.0},                 # an object
    [],
])
def test_a_control_point_without_three_coordinates_is_skipped(tmp_path, position):
    """Skipped, not crashed on: one malformed point must not cost the patient
    the whole file."""
    point = {"label": "A", "positionStatus": "defined"}
    if position is not None:
        point["position"] = position
    source = tmp_path / "P1_lm.mrk.json"
    source.write_text(json.dumps({"markups": [{"controlPoints": [
        point,
        {"label": "B", "position": [10.0, 0.0, 0.0], "positionStatus": "defined"},
    ]}]}))
    out = tmp_path / "out.mrk.json"

    assert pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (4.0, 0.0, 0.0)), str(out)) == 1
    kept = json.loads(out.read_text())["markups"][0]["controlPoints"]
    assert kept[0].get("position") == position, "a point that could not move was altered"
    assert kept[1]["position"] == [6.0, 0.0, 0.0]


@pytest.mark.parametrize("status", ["undefined", "preview", ""])
def test_a_point_whose_position_is_not_defined_is_left_where_it_is(tmp_path, status):
    source = tmp_path / "P1_lm.mrk.json"
    source.write_text(json.dumps({"markups": [{"controlPoints": [
        {"label": "A", "position": [1.0, 2.0, 3.0], "positionStatus": status}]}]}))
    out = tmp_path / "out.mrk.json"

    assert pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (4.0, 0.0, 0.0)), str(out)) == 0
    assert _points(out) == [[1.0, 2.0, 3.0]]


def test_a_point_with_no_position_status_at_all_still_moves(tmp_path):
    """"Defined" is the default Slicer's own markups schema declares, and ASO
    and AREG read a control point without consulting the field. Requiring it
    left a point from any other writer sitting at its old coordinates inside a
    file reported as transformed."""
    source = tmp_path / "P1_lm.mrk.json"
    source.write_text(json.dumps({"markups": [{"controlPoints": [
        {"label": "A", "position": [10.0, 0.0, 0.0]}]}]}))
    out = tmp_path / "out.mrk.json"

    assert pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (4.0, 0.0, 0.0)), str(out)) == 1
    assert _points(out) == [[6.0, 0.0, 0.0]]


# ---------------------------------------------------------------------------
# Everything that is not a coordinate
# ---------------------------------------------------------------------------

def test_the_labels_and_descriptions_survive_untouched(tmp_path, slicer_markups):
    """A landmark's label is how a clinician knows which point it is, and its
    description is where the caveat about how it was placed travels."""
    source = slicer_markups(tmp_path / "P1_lm.mrk.json",
                            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    out = tmp_path / "out.mrk.json"

    pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (1.0, 0.0, 0.0)), str(out))

    points = json.loads(out.read_text())["markups"][0]["controlPoints"]
    assert [point["label"] for point in points] == ["Ba0", "Ba1"]
    assert [point["description"] for point in points] == [
        "placed at scale 0", "placed at scale 1"]
    assert [point["id"] for point in points] == ["1", "2"]


def test_the_display_node_is_not_switched_off(tmp_path, slicer_markups):
    """`display.visibility: false` is the defect both original CLIs shipped:
    Slicer loads the file, builds the node and draws nothing. A file that
    arrives visible leaves visible."""
    source = slicer_markups(tmp_path / "P1_lm.mrk.json")
    out = tmp_path / "out.mrk.json"

    pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (1.0, 0.0, 0.0)), str(out))

    display = json.loads(out.read_text())["markups"][0]["display"]
    assert display["visibility"] is True
    assert display["opacity"] == 1.0
    assert display["glyphScale"] == 2.0


def test_the_per_point_visibility_and_locks_survive(tmp_path, slicer_markups):
    source = slicer_markups(tmp_path / "P1_lm.mrk.json")
    out = tmp_path / "out.mrk.json"

    pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (1.0, 0.0, 0.0)), str(out))

    point = json.loads(out.read_text())["markups"][0]["controlPoints"][0]
    assert point["visibility"] is True
    assert point["selected"] is True
    assert point["locked"] is True
    assert point["orientation"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_the_schema_and_the_coordinate_system_survive(tmp_path, slicer_markups):
    """A markups file without its `@schema` loads but is not recognised by
    version-aware readers, and a file that lost `coordinateSystem` is a file
    whose coordinates mean nothing."""
    source = slicer_markups(tmp_path / "P1_lm.mrk.json")
    out = tmp_path / "out.mrk.json"

    pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (1.0, 0.0, 0.0)), str(out))

    document = json.loads(out.read_text())
    assert document["@schema"].endswith("markups-schema-v1.0.0.json#")
    assert document["markups"][0]["coordinateSystem"] == "LPS"
    assert document["markups"][0]["type"] == "Fiducial"
    assert document["markups"][0]["labelFormat"] == "%N-%d"


def test_only_the_coordinates_change(tmp_path, slicer_markups):
    """Everything else, key by key, byte for byte."""
    source = slicer_markups(tmp_path / "P1_lm.mrk.json", [(1.0, 2.0, 3.0)])
    out = tmp_path / "out.mrk.json"

    pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (1.0, 0.0, 0.0)), str(out))

    before = json.loads(source.read_text())
    after = json.loads(out.read_text())
    for document in (before, after):
        for group in document["markups"]:
            for point in group["controlPoints"]:
                point.pop("position")
    assert after == before


def test_the_source_file_is_not_modified(tmp_path, slicer_markups):
    """The input is the clinician's own file; a run must be able to be repeated."""
    source = slicer_markups(tmp_path / "P1_lm.mrk.json", [(1.0, 2.0, 3.0)])
    original = source.read_text()

    pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (1.0, 0.0, 0.0)),
        str(tmp_path / "out.mrk.json"))

    assert source.read_text() == original


def test_the_count_returned_is_the_number_of_points_moved(tmp_path):
    source = tmp_path / "P1_lm.mrk.json"
    source.write_text(json.dumps({"markups": [{"controlPoints": [
        {"label": "A", "position": [1.0, 0.0, 0.0], "positionStatus": "defined"},
        {"label": "B", "position": [2.0, 0.0, 0.0], "positionStatus": "undefined"},
        {"label": "C", "position": [3.0, 0.0, 0.0], "positionStatus": "defined"},
        {"label": "D", "positionStatus": "defined"},
    ]}]}))

    assert pipeline.apply_to_landmarks(
        str(source), sitk.TranslationTransform(3, (1.0, 0.0, 0.0)),
        str(tmp_path / "out.mrk.json")) == 2


@pytest.mark.parametrize("name,expected", [
    ("P1_lm.mrk.json", True),
    ("P1_lm.MRK.JSON", True),
    ("P1_lm_Pred.mrk.json", True),
    ("P1.json", False),
    ("P1_mrk.json", False),
    ("P1.nii.gz", False),
])
def test_what_counts_as_a_landmark_file(name, expected):
    """`.json` alone is not one: ALI wrote `.json` for byte-identical content
    and Slicer associates only the first spelling with a markups node."""
    assert pipeline.is_landmark_file(name) is expected
