"""Reading landmark files, and matching one to the mesh it belongs to.

Two shapes are read because upstream's own test set carries both -- `.mrk.json`
for the CBCT side, `.json` for the intraoral one -- and neither side is
declared canonical.

The matching is where this tool can silently produce a wrong answer. A mesh
paired with another patient's landmarks still yields a rigid transform, still
writes a mesh and still reports "ok"; the only sign is that the intraoral scan
is in someone else's skull.
"""

import os

import pytest

from conftest import (
    LOWER_LABELS, UPPER_LABELS, moved, write_flat_landmarks, write_markups,
)
from sadt_areg_ioscbct import dispatch, pipeline
from sadt_areg_common.errors import ToolInputError


# ---------------------------------------------------------------------------
# read_landmarks
# ---------------------------------------------------------------------------

def test_a_slicer_markups_file_is_read(tmp_path):
    path = write_markups(tmp_path / "a.mrk.json", UPPER_LABELS)
    assert pipeline.read_landmarks(path) == UPPER_LABELS


def test_the_plain_shape_is_read_too(tmp_path):
    """`{label: [x, y, z]}` at the top level, which is what upstream's
    intraoral side ships."""
    path = write_flat_landmarks(tmp_path / "a.json", UPPER_LABELS)
    assert pipeline.read_landmarks(path) == UPPER_LABELS


def test_positions_come_back_as_floats(tmp_path):
    """They arrive as whatever JSON held -- ints, or strings from a hand-made
    file -- and the alignment builds a numpy array out of them."""
    path = write_flat_landmarks(tmp_path / "a.json", {"UR1O": ["1", 2, 3.5]})
    assert pipeline.read_landmarks(path) == {"UR1O": [1.0, 2.0, 3.5]}


def test_a_control_point_missing_its_label_or_position_is_skipped(tmp_path):
    """A markups file edited by hand is a real input, and an unlabelled point
    would otherwise key the dictionary on None."""
    import json

    content = {
        "markups": [
            {
                "controlPoints": [
                    {"label": "UR1O", "position": [1.0, 2.0, 3.0]},
                    {"label": "", "position": [4.0, 5.0, 6.0]},
                    {"label": "UL1O"},
                ]
            }
        ]
    }
    path = tmp_path / "a.mrk.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    assert pipeline.read_landmarks(str(path)) == {"UR1O": [1.0, 2.0, 3.0]}


def test_a_json_file_that_holds_no_landmarks_reads_as_empty(tmp_path):
    path = write_flat_landmarks(tmp_path / "a.json", {})
    assert pipeline.read_landmarks(path) == {}


def test_a_key_that_is_not_a_three_vector_is_not_a_landmark(tmp_path):
    """The flat fallback iterates every top-level key, so a file carrying
    metadata beside its points must not turn the metadata into a landmark."""
    import json

    path = tmp_path / "a.json"
    path.write_text(
        json.dumps({"UR1O": [1.0, 2.0, 3.0], "version": "2", "bounds": [0, 1]}),
        encoding="utf-8",
    )
    assert pipeline.read_landmarks(str(path)) == {"UR1O": [1.0, 2.0, 3.0]}


# ---------------------------------------------------------------------------
# shared_landmarks
# ---------------------------------------------------------------------------

def test_the_two_sides_are_intersected_rather_than_assumed_equal():
    """The two modalities are landmarked by different networks, and one point
    missing on one side used to be an IndexError three frames down instead of
    a line in a report."""
    cbct = dict(moved(UPPER_LABELS))
    cbct.pop("UL6O")
    cbct["LR1O"] = [0.0, 0.0, -30.0]

    moving, fixed, used, dropped = pipeline.shared_landmarks(UPPER_LABELS, cbct)
    assert used == sorted(set(UPPER_LABELS) - {"UL6O"})
    assert dropped == ["LR1O", "UL6O"]
    assert moving.shape == fixed.shape == (len(used), 3)


def test_the_two_arrays_are_in_the_same_order():
    """They are fed to `vtkLandmarkTransform` as source and target: one row out
    of step pairs a canine with a molar and the fit is quietly wrong."""
    moving, fixed, used, _dropped = pipeline.shared_landmarks(
        UPPER_LABELS, moved(UPPER_LABELS)
    )
    for row, label in enumerate(used):
        assert list(moving[row]) == UPPER_LABELS[label]
        assert list(fixed[row]) == moved(UPPER_LABELS)[label]


def test_fewer_than_three_shared_points_names_both_sides():
    """An alignment needs three, and the message has to say what each side
    actually offered -- otherwise the fix is guesswork."""
    with pytest.raises(ToolInputError) as raised:
        pipeline.shared_landmarks(
            {"UR1O": [0, 0, 0], "UR3O": [1, 0, 0]}, {"UR1O": [0, 0, 0]}
        )
    message = str(raised.value)
    assert "share only 1 landmark(s)" in message
    assert "UR3O" in message and "UR1O" in message


def test_exactly_three_shared_points_is_enough():
    labels = {"UR1O": [0, 0, 0], "UR3O": [10, 0, 0], "UL1O": [0, 10, 0]}
    _moving, _fixed, used, _dropped = pipeline.shared_landmarks(labels, moved(labels))
    assert used == ["UL1O", "UR1O", "UR3O"]


# ---------------------------------------------------------------------------
# Finding the files
# ---------------------------------------------------------------------------

def test_both_landmark_spellings_are_collected(tmp_path):
    write_markups(tmp_path / "lm" / "P001_T2_U_lm_Pred.mrk.json", UPPER_LABELS)
    write_flat_landmarks(tmp_path / "lm" / "P001_T2_L_lm_Pred.json", LOWER_LABELS)

    found = dispatch._landmarks_by_jaw(str(tmp_path / "lm"))
    assert set(found) == {"P001_T2_U_lm_Pred.mrk.json", "P001_T2_L_lm_Pred.json"}


def test_files_are_keyed_by_their_path_relative_to_the_folder(tmp_path):
    """ALI mirrors the input's tree, so two sites' `P1_U_lm_Pred.mrk.json` are
    two files. Keyed by base name the second silently replaced the first."""
    write_markups(tmp_path / "lm" / "siteA" / "P1_U_lm_Pred.mrk.json", UPPER_LABELS)
    write_markups(tmp_path / "lm" / "siteB" / "P1_U_lm_Pred.mrk.json", UPPER_LABELS)

    found = dispatch._landmarks_by_jaw(str(tmp_path / "lm"))
    assert set(found) == {
        os.path.join("siteA", "P1_U_lm_Pred.mrk.json"),
        os.path.join("siteB", "P1_U_lm_Pred.mrk.json"),
    }


def test_a_file_that_is_not_json_is_not_a_landmark_file(tmp_path):
    write_markups(tmp_path / "lm" / "P001_U_lm_Pred.mrk.json", UPPER_LABELS)
    (tmp_path / "lm" / "P001_U.vtk").write_text("mesh")
    (tmp_path / "lm" / "notes.txt").write_text("hello")

    assert list(dispatch._landmarks_by_jaw(str(tmp_path / "lm"))) == [
        "P001_U_lm_Pred.mrk.json"
    ]


def test_an_empty_landmark_file_is_not_listed(tmp_path):
    """`register` refuses the run when a folder holds nothing usable, and a
    file with no points is nothing usable."""
    write_flat_landmarks(tmp_path / "lm" / "empty.json", {})
    assert dispatch._landmarks_by_jaw(str(tmp_path / "lm")) == {}


def test_a_missing_folder_is_no_landmarks_rather_than_a_crash(tmp_path):
    assert dispatch._landmarks_by_jaw(str(tmp_path / "nowhere")) == {}
    assert dispatch._landmarks_by_jaw("") == {}
    assert dispatch._landmarks_by_jaw(None) == {}


# ---------------------------------------------------------------------------
# Matching a file to a mesh
# ---------------------------------------------------------------------------

def test_the_jaw_token_decides_never_the_sort_order():
    """Pairing by position is how an upper mesh gets registered against a lower
    arch's points -- and the two files sort in whichever order their names
    happen to."""
    candidates = {
        "P001_T2_L_lm_Pred.mrk.json": LOWER_LABELS,
        "P001_T2_U_lm_Pred.mrk.json": UPPER_LABELS,
    }
    assert dispatch._match_landmarks("P001_T2_U.vtk", candidates) == UPPER_LABELS
    assert dispatch._match_landmarks("P001_T2_L.vtk", candidates) == LOWER_LABELS


@pytest.mark.parametrize("token", ["U", "u", "Upper", "L", "lower"])
def test_the_jaw_token_is_matched_whichever_way_it_is_spelled(token):
    """Case does not matter; the WORD does. ALI names its output after the mesh
    it read, so the two sides always carry the same word -- which is why this
    is a set intersection over `u`/`upper`/`l`/`lower` and not the wider
    synonym table ASO uses."""
    candidates = {f"P1_{token}_lm.json": UPPER_LABELS, "P1_other_lm.json": LOWER_LABELS}
    assert dispatch._match_landmarks(f"P1_{token.upper()}.vtk", candidates) == UPPER_LABELS


def test_a_jaw_written_two_different_ways_does_not_match_by_accident():
    """`_U` against `_Upper` is not a match here, and the refusal that follows
    names the mesh's jaw -- which beats silently pairing an upper mesh with
    whichever file sorted first."""
    candidates = {"P1_Upper_lm.json": UPPER_LABELS, "P1_Lower_lm.json": LOWER_LABELS}
    assert dispatch._match_landmarks("P1_U.vtk", candidates) == {}


def test_one_file_with_no_jaw_in_its_name_covers_every_jaw():
    """A CBCT covers both arches in one volume, so its landmark file has no jaw
    to name. The labels carry it instead -- `UR1O` against `LR1O` -- and
    `shared_landmarks` intersects."""
    both = dict(UPPER_LABELS, **LOWER_LABELS)
    candidates = {"P_0001_T2_lm_Pred.mrk.json": both}
    assert dispatch._match_landmarks("P001_T2_U.vtk", candidates) == both
    assert dispatch._match_landmarks("P001_T2_L.vtk", candidates) == both


def test_two_unlabelled_files_are_not_guessed_between():
    """One is a fallback; two is a question this cannot answer, and answering
    it by sort order is what the jaw rule exists to prevent."""
    candidates = {"a_lm.json": UPPER_LABELS, "b_lm.json": LOWER_LABELS}
    assert dispatch._match_landmarks("P001_T2_U.vtk", candidates) == {}


def test_a_jaw_match_beats_an_unlabelled_file():
    # The unlabelled one sorts FIRST, so a loop that took the first candidate
    # it could use would take it.
    candidates = {
        "A_0001_T2_lm_Pred.mrk.json": LOWER_LABELS,
        "P001_T2_U_lm_Pred.mrk.json": UPPER_LABELS,
    }
    assert sorted(candidates)[0].startswith("A_")
    assert dispatch._match_landmarks("P001_T2_U.vtk", candidates) == UPPER_LABELS


def test_no_candidate_at_all_is_no_match():
    assert dispatch._match_landmarks("P001_T2_U.vtk", {}) == {}


# ---------------------------------------------------------------------------
# Narrowing to the patient, before the jaw is looked at
# ---------------------------------------------------------------------------

def test_a_patients_own_files_are_the_only_ones_offered():
    """The defect this exists to remove: every ALI_IOS file carries a `_U`
    token, `sorted()` puts P1's first, and the first jaw match won -- so the
    second patient's mesh was registered against the FIRST patient's
    landmarks, and the report said "ok"."""
    candidates = {
        "P001_T2_U_lm_Pred.mrk.json": UPPER_LABELS,
        "P002_T2_U_lm_Pred.mrk.json": LOWER_LABELS,
    }
    assert dispatch._for_patient(candidates, "2", sole_patient=False) == {
        "P002_T2_U_lm_Pred.mrk.json": LOWER_LABELS
    }


def test_a_patient_with_no_file_of_its_own_gets_nothing_rather_than_a_neighbours():
    candidates = {"P001_T2_U_lm_Pred.mrk.json": UPPER_LABELS}
    assert dispatch._for_patient(candidates, "2", sole_patient=False) == {}


def test_a_single_patient_batch_keeps_the_looser_rule():
    """There the two sides cannot be confused, and the published reference
    files -- `Upper_gold.vtk`, no digits at all -- would otherwise stop
    matching anything."""
    candidates = {"Upper_gold_lm.json": UPPER_LABELS}
    assert dispatch._for_patient(candidates, "Upper_gold", sole_patient=True) == candidates
    assert dispatch._for_patient(candidates, "Upper_gold", sole_patient=False) == {}
