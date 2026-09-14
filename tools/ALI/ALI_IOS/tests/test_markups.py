"""The Slicer markups file this tool hands back.

`sadt_ali_common.markups` is shared with ALI_CBCT precisely so the two cannot
disagree about it, which is a reason to test it from BOTH sides rather than
from whichever one happens to have a suite: a change made for the CBCT engine
lands in this tool's output too.

The one value here that is not cosmetic is `display.visibility`. Both upstream
CLIs wrote `false`, which switches the markups DISPLAY node off: Slicer loads
the file, builds the node, lists it in the Markups module -- and draws nothing.
Invisible inside the old Slicer module, which loaded the nodes itself; fatal
for anyone opening a returned archive.
"""

import json
import os

import numpy as np
import pytest

from sadt_ali_common import markups


def read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write(tmp_path, landmarks, descriptions=None, name="scan_lm_Pred.mrk.json"):
    destination = str(tmp_path / name)
    return read(markups.write(landmarks, destination, descriptions=descriptions))


def test_the_display_node_is_visible(tmp_path):
    """The headline correction. `false` here is a file Slicer opens and draws
    nothing from, which reads as "the tool returned no landmarks"."""
    content = write(tmp_path, {"UR1O": (1.0, 2.0, 3.0)})
    assert content["markups"][0]["display"]["visibility"] is True


def test_each_control_point_is_visible_too(tmp_path):
    """Independent of the node's own visibility: a point can be visible in a
    node that is not displayed, which is exactly why the node-level flag went
    unnoticed."""
    content = write(tmp_path, {"UR1O": (1.0, 2.0, 3.0), "UR1MB": (4.0, 5.0, 6.0)})
    for point in content["markups"][0]["controlPoints"]:
        assert point["visibility"] is True
        assert point["positionStatus"] == "defined"


def test_slice_projection_stays_off(tmp_path):
    """A preference, unlike `visibility`, and deliberately the other way: with
    a three-figure point count, projecting each one onto its neighbouring
    slices crowds the very view used to judge placement."""
    content = write(tmp_path, {"UR1O": (0.0, 0.0, 0.0)})
    assert content["markups"][0]["display"]["sliceProjection"] is False


def test_the_coordinate_system_is_declared_lps(tmp_path):
    """Both engines produce LPS and both original CLIs declared it. Stated once
    in the shared writer so a future engine cannot quietly write RAS into a
    file that claims otherwise."""
    content = write(tmp_path, {"UR1O": (0.0, 0.0, 0.0)})
    assert content["markups"][0]["coordinateSystem"] == "LPS"
    assert markups.COORDINATE_SYSTEM == "LPS"


def test_the_extension_is_the_one_slicer_associates_with_markups():
    """The IOS CLI wrote `.json` for byte-identical content, and Slicer only
    associates `.mrk.json` with a markups node -- so half of ALI's output had
    to be imported by hand."""
    assert markups.MARKUPS_EXTENSION == ".mrk.json"


def test_the_schema_url_is_written(tmp_path):
    """A markups file without it loads, but is not recognised by version-aware
    readers."""
    content = write(tmp_path, {"UR1O": (0.0, 0.0, 0.0)})
    assert content["@schema"].endswith("markups-schema-v1.0.0.json#")


def test_numpy_coordinates_survive_the_json_round_trip(tmp_path):
    """Positions arrive as numpy scalars -- `upscale` returns an ndarray -- and
    `json.dump` cannot serialize those. The failure would land at the very end
    of a run, after all the inference."""
    position = np.array([1.5, -2.5, 3.5], dtype=np.float32)
    content = write(tmp_path, {"UR1O": position})
    written = content["markups"][0]["controlPoints"][0]["position"]
    assert written == pytest.approx([1.5, -2.5, 3.5])
    assert all(isinstance(value, float) for value in written)


def test_a_caveat_travels_with_the_point_it_belongs_to(tmp_path):
    """A degraded landmark looks exactly like a good one in the file, and
    whoever opens it is the one who has to know which to review -- so the note
    is in the control point, not only in the report."""
    content = write(
        tmp_path,
        {"L0MG": (0.0, 0.0, 0.0), "LR1MG": (1.0, 1.0, 1.0)},
        descriptions={"L0MG": "forced (confidence 0.120)"},
    )
    by_label = {p["label"]: p for p in content["markups"][0]["controlPoints"]}
    assert by_label["L0MG"]["description"] == "forced (confidence 0.120)"
    # And a point that needed no caveat carries an empty one rather than None,
    # which Slicer renders as the string "None".
    assert by_label["LR1MG"]["description"] == ""


def test_control_points_keep_the_order_they_were_placed_in(tmp_path):
    landmarks = {"UR1O": (0, 0, 0), "UR1MB": (1, 1, 1), "UR1DB": (2, 2, 2)}
    content = write(tmp_path, landmarks)
    assert [p["label"] for p in content["markups"][0]["controlPoints"]] == list(landmarks)
    assert [p["id"] for p in content["markups"][0]["controlPoints"]] == ["1", "2", "3"]


def test_writing_creates_the_folder_the_output_tree_needs(tmp_path):
    """The destination mirrors the input's own tree, so the parent directory
    routinely does not exist yet."""
    destination = str(tmp_path / "out" / "siteA" / "nested" / "scan_lm_Pred.mrk.json")
    markups.write({"UR1O": (0.0, 0.0, 0.0)}, destination)
    assert os.path.isfile(destination)


def test_an_empty_landmark_set_still_writes_a_readable_file(tmp_path):
    content = write(tmp_path, {})
    assert content["markups"][0]["controlPoints"] == []
