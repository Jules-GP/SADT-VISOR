"""ALI_IOS: the catalog it publishes, and the weights it recognises.

These moved out of ALI_CBCT's test file when ALI became two tools. They were
still there, importing `sadt_ali.ios`, which no virtualenv has provided since
the split -- so ALI_CBCT's whole suite failed to collect and ALI_IOS had no
tests at all. Found by running each tool's suite in its own interpreter.
"""

import os
import typing

import pytest

from sadt_ali_ios import catalog, engine
from sadt_ali_ios.errors import ToolInputError
from sadt_ali_ios import run


def _choices(argument):
    """The `Literal` options `run()` publishes for one argument."""
    hint = typing.get_type_hints(run)[argument]
    if typing.get_origin(hint) is list:
        hint = typing.get_args(hint)[0]
    return list(typing.get_args(hint))


# Copied from ALI_CBCT's suite rather than shared: the two tools are
# separate packages with separate virtualenvs, and CONTRIBUTING.md says a
# test helper is duplicated rather than given a package of its own.
def write_surface(path, labelled=True):
    """A minimal .vtk polydata, optionally carrying a tooth-label array."""
    vtk = pytest.importorskip("vtk")

    points = vtk.vtkPoints()
    for coordinates in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)):
        points.InsertNextPoint(*coordinates)

    polys = vtk.vtkCellArray()
    for triangle in ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)):
        polys.InsertNextCell(3)
        for point_id in triangle:
            polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)

    if labelled:
        labels = vtk.vtkIntArray()
        labels.SetName("Universal_ID")
        for value in (8, 8, 8, 8):
            labels.InsertNextValue(value)
        surface.GetPointData().AddArray(labels)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)

def write_ios_bundle(root, names):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"fake checkpoint")
    return str(root)

def test_ios_offers_only_landmark_types_a_model_predicts():
    """R, RIP and OIP were selectable in the Slicer UI and predicted by
    nothing: no network produced them and no label table contained them.
    Ticking them did literally nothing."""
    offered = {lm_type for types in catalog.NETWORKS.values() for lm_type in types}
    assert offered == {"O", "MB", "DB", "CL", "CB", "MG"}
    assert not offered & {"R", "RIP", "OIP"}


def test_ios_tooth_numbering_matches_the_shipped_label_tables():
    assert catalog.UNIVERSAL_NUMBERS["Upper"]["UL7"] == 15
    assert catalog.UNIVERSAL_NUMBERS["Upper"]["UR7"] == 2
    assert catalog.UNIVERSAL_NUMBERS["Lower"]["LL7"] == 18
    assert catalog.UNIVERSAL_NUMBERS["Lower"]["LR7"] == 31
    # Tooth 8 is UR1: the occlusal network's three channels, in channel order.
    assert catalog.LABELS["O"]["8"] == ["UR1O", "UR1MB", "UR1DB"]
    assert catalog.LABELS["C"]["8"] == ["UR1CL", "UR1CB"]


def test_ios_network_codes_from_a_selection():
    assert catalog.network_codes(["Occlusal"]) == ("O",)
    assert catalog.network_codes(["O"]) == ("O",)
    assert catalog.network_codes(None) == catalog.NETWORK_CODES
    with pytest.raises(ValueError, match="Occlusal"):
        catalog.network_codes(["Buccal"])


def test_the_published_networks_are_the_catalogs_own():
    assert _choices("networks") == list(catalog.NETWORK_NAMES)


def test_ios_weight_discovery_reads_the_published_names(tmp_path):
    """The real ALIDDM bundle: Upper_O_model.pth, Lower_C_model.pth, ..."""
    bundle = write_ios_bundle(
        tmp_path / "bundle",
        ["Upper_O_model.pth", "Lower_O_model.pth", "Upper_C_model.pth", "Lower_C_model.pth"],
    )
    weights, unrecognized = engine.discover_weights(bundle)

    assert weights["O"].keys() == {"Upper", "Lower"}
    assert weights["C"].keys() == {"Upper", "Lower"}
    assert unrecognized == []


def test_an_ios_checkpoint_with_no_jaw_token_is_reported_not_assumed_upper(tmp_path):
    """The original treated every file not containing "Lower" as upper-jaw
    weights, so a bundle missing its mandibular model quietly predicted the
    lower arch with the maxillary one."""
    bundle = write_ios_bundle(tmp_path / "bundle", ["model_O.pth", "Upper_C_model.pth"])
    weights, unrecognized = engine.discover_weights(bundle)

    assert unrecognized == ["model_O.pth"]
    assert "O" not in weights
    assert weights["C"] == {"Upper": str(tmp_path / "bundle" / "Upper_C_model.pth")}


def test_a_mesh_without_tooth_labels_names_the_tool_that_makes_them(tmp_path):
    """The handoff `ALILogic.ensure_segmented()` used to make in-process.

    Tools do not call each other any more, so this cannot segment the mesh
    itself -- but it can say exactly what to run, which is the difference
    between a fixable request and "no known tooth number is present".
    """
    labelled = write_surface(tmp_path / "in" / "good.vtk", labelled=True)
    raw = write_surface(tmp_path / "in" / "raw.vtk", labelled=False)

    with pytest.raises(ToolInputError) as raised:
        engine.require_labels([(labelled, "good.vtk"), (raw, "raw.vtk")])

    message = str(raised.value)
    assert "Crown_Seg" in message
    assert "1 of 2" in message
    # The array names it looked for, so the fix is actionable without reading
    # the source.
    assert "Universal_ID" in message


def test_a_fully_labelled_batch_passes_the_check(tmp_path):
    mesh = write_surface(tmp_path / "in" / "arch.vtk", labelled=True)
    assert engine.require_labels([(mesh, "arch.vtk")]) is None


def test_mucogingival_is_offered_but_not_on_by_default():
    """One point per lower tooth on the gingival margin, wanted by a mandible
    registration and by nobody asking for crown landmarks. On by default would
    add a third pass over every mesh of every existing request."""
    import inspect

    assert catalog.NETWORK_NAMES["Mucogingival"] == "MG"
    assert "Mucogingival" in _choices("networks")
    assert inspect.signature(run).parameters["networks"].default == [
        "Occlusal", "Cervical"
    ]


def test_mucogingival_runs_on_the_mandible_only():
    """It was trained on the mandible alone, so a maxilla is not a missing
    model -- it is a question the network cannot be asked."""
    assert catalog.NETWORK_JAWS["MG"] == ("Lower",)
    # The other two are unrestricted, and must stay that way.
    assert "O" not in catalog.NETWORK_JAWS
    assert "C" not in catalog.NETWORK_JAWS


def test_the_mucogingival_names_are_positional_not_derived():
    """Six MG output names collide with the TRAINING name of a DIFFERENT tooth
    -- LR1MG is the training name of tooth 25 and the output name of tooth 26 --
    because tooth 25 carries the midline name L0MG and shifts the right side by
    one. Deriving `<tooth><type>` here would mislabel half the arch."""
    labels = catalog.LABELS["MG"]

    assert labels["19"] == ["LL6MG"]      # first trained tooth
    assert labels["25"] == ["L0MG"]       # the midline, not "LR1MG"
    assert labels["26"] == ["LR1MG"]      # shifted by one against the numbers
    assert labels["31"] == ["LR6MG"]
    # Tooth 18 was excluded from training and has no MG label at all.
    assert "18" not in labels
    assert len(labels) == 13 == len(catalog.MG_TEETH)


def test_every_mucogingival_tooth_has_an_aim_offset():
    """The cameras aim at the landmark's expected position rather than at a
    flat drop below the tooth centre, which only ever matched the incisors: on
    the molars the landmark is ~0.15 further buccal and fell outside the render
    entirely. A tooth with no offset would be back to that."""
    assert set(catalog.MG_AIM_OFFSET) == set(catalog.MG_TEETH)
    for tooth, offset in catalog.MG_AIM_OFFSET.items():
        assert len(offset) == 3, tooth
        # Below the crown, always: the gingival margin is under it.
        assert offset[2] < 0, tooth


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def test_progress_writes_one_line_per_event_and_nothing_when_unset(tmp_path, monkeypatch):
    """The engine's own loop cannot run here -- it needs pytorch3d, which
    publishes no usable wheel -- so what is tested is this package's copy of
    the helper the loop calls, on the contract every tool's copy holds to.
    """
    import json

    from sadt_ali_ios import progress

    monkeypatch.delenv(progress.VARIABLE, raising=False)
    progress.report(1, 4, "mesh")  # a no-op, and above all not an exception

    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv(progress.VARIABLE, str(events_file))
    progress.report(3, 4, "mesh")

    assert json.loads(events_file.read_text()) == {
        "fraction": 0.5, "message": "mesh 3 of 4",
    }


def test_the_engine_reports_the_position_it_already_logs():
    """The counter was there and went only to the server's log. The rule the
    log line states -- position in the batch, never the mesh's name -- is why
    the progress call could take it unchanged."""
    source = open(engine.__file__, encoding="utf-8").read()
    assert 'progress.report(mesh_index, len(meshes), "mesh")' in source
