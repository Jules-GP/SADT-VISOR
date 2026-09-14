"""What `dispatch.detect` accepts, what it refuses, and how it says so.

Before the split, detection chose WHICH engine ran. It no longer does: this
tool is the intraoral engine, so the only question is whether the caller sent
surfaces -- and the interesting half is the refusals, because silently
processing the meshes of a mixed folder and dropping the volumes is the failure
that looks like success.

Every message asserted here reaches the client verbatim as a 422, so each test
checks it NAMES the thing the caller has to change.
"""

import os

import pytest

from conftest import write_surface
from sadt_ali_common.discovery import IOS, WORK_DIRNAME
from sadt_ali_ios import dispatch
from sadt_ali_ios.errors import ToolInputError


def detect(path, tmp_path):
    return dispatch.detect(str(path), str(tmp_path / "work"))


# ---------------------------------------------------------------------------
# One file
# ---------------------------------------------------------------------------

def test_a_single_vtk_is_one_scan_keyed_by_its_own_name(tmp_path):
    mesh = write_surface(tmp_path / "in" / "arch.vtk")
    detected = detect(mesh, tmp_path)
    assert detected.mode == IOS
    assert detected.scans == [(mesh, "arch.vtk")]


def test_a_single_stl_is_accepted(tmp_path):
    """`.stl` was counted by the Slicer UI and then never discovered by the
    CLI, which globbed for `.vtk` alone: accepted, and silently ignored."""
    mesh = tmp_path / "in" / "arch.stl"
    os.makedirs(mesh.parent, exist_ok=True)
    mesh.write_bytes(b"solid empty\nendsolid empty\n")
    detected = detect(mesh, tmp_path)
    assert detected.scans == [(str(mesh), "arch.stl")]


def test_the_extension_is_matched_case_insensitively(tmp_path):
    """A scanner that writes `ARCH.VTK` sends the same file. Refusing it would
    be a 422 about a difference that is not one."""
    mesh = write_surface(tmp_path / "in" / "ARCH.VTK")
    assert detect(mesh, tmp_path).scans == [(mesh, "ARCH.VTK")]


def test_a_single_volume_is_refused_by_name_and_names_the_other_tool(tmp_path):
    """The whole reason the split moved this question out of the run: a caller
    who sent CBCT data has to be told which tool takes it, not handed an empty
    result."""
    volume = tmp_path / "in" / "patient01.nii.gz"
    os.makedirs(volume.parent, exist_ok=True)
    volume.write_bytes(b"not really a volume")

    with pytest.raises(ToolInputError) as raised:
        detect(volume, tmp_path)
    message = str(raised.value)
    assert "patient01.nii.gz" in message
    assert "ALI_CBCT" in message


@pytest.mark.parametrize("name", ["scan.nii", "scan.nrrd", "scan.gipl.gz", "scan.nrrd.gz"])
def test_every_volume_spelling_is_recognised_as_a_volume(tmp_path, name):
    """Compound extensions included: `.nrrd.gz` must not fall through to "not
    an intraoral surface", which sends the caller looking at the wrong thing."""
    volume = tmp_path / "in" / name
    os.makedirs(volume.parent, exist_ok=True)
    volume.write_bytes(b"x")
    with pytest.raises(ToolInputError, match="ALI_CBCT"):
        detect(volume, tmp_path)


def test_an_unrecognised_single_file_names_the_file_and_the_extensions(tmp_path):
    other = tmp_path / "in" / "notes.txt"
    os.makedirs(other.parent, exist_ok=True)
    other.write_text("hello")

    with pytest.raises(ToolInputError) as raised:
        detect(other, tmp_path)
    message = str(raised.value)
    assert "notes.txt" in message
    assert ".vtk" in message and ".stl" in message


def test_a_missing_input_names_the_path_it_was_given(tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere"):
        detect(tmp_path / "nowhere", tmp_path)


def test_an_error_message_never_carries_the_server_side_directory(tmp_path):
    """A 422 body travels to the client. The server's own layout does not."""
    other = tmp_path / "secret_job_dir" / "notes.txt"
    os.makedirs(other.parent, exist_ok=True)
    other.write_text("hello")

    with pytest.raises(ToolInputError) as raised:
        detect(other, tmp_path)
    assert "secret_job_dir" not in str(raised.value)


# ---------------------------------------------------------------------------
# A folder
# ---------------------------------------------------------------------------

def test_a_folder_is_searched_recursively_and_keeps_its_tree(tmp_path):
    root = tmp_path / "in"
    write_surface(root / "siteA" / "p1.vtk")
    write_surface(root / "siteB" / "nested" / "p2.vtk")

    detected = detect(root, tmp_path)
    assert [key for _path, key in detected.scans] == [
        os.path.join("siteA", "p1.vtk"),
        os.path.join("siteB", "nested", "p2.vtk"),
    ]


def test_two_meshes_of_the_same_name_get_different_keys(tmp_path):
    """Keying by BASE NAME made two patients called `scan.vtk` in different
    folders overwrite each other twice over -- in the working dictionary and
    again in the flat output folder."""
    root = tmp_path / "in"
    write_surface(root / "patientA" / "scan.vtk")
    write_surface(root / "patientB" / "scan.vtk")

    keys = [key for _path, key in detect(root, tmp_path).scans]
    assert len(set(keys)) == 2


def test_discovery_is_sorted_so_a_batch_runs_in_a_stable_order(tmp_path):
    root = tmp_path / "in"
    for name in ("zeta.vtk", "alpha.vtk", "mid.vtk"):
        write_surface(root / name)
    keys = [key for _path, key in detect(root, tmp_path).scans]
    assert keys == ["alpha.vtk", "mid.vtk", "zeta.vtk"]


def test_a_folder_of_volumes_only_is_refused_with_a_count(tmp_path):
    root = tmp_path / "in"
    os.makedirs(root, exist_ok=True)
    for name in ("a.nii.gz", "b.nii.gz"):
        (root / name).write_bytes(b"x")

    with pytest.raises(ToolInputError) as raised:
        detect(root, tmp_path)
    message = str(raised.value)
    assert "2 CBCT scan(s)" in message
    assert "ALI_CBCT" in message


def test_a_mixed_folder_is_refused_rather_than_half_processed(tmp_path):
    """Processing the meshes and dropping the volumes is the failure that looks
    like success, so both counts are named and both tools are."""
    root = tmp_path / "in"
    os.makedirs(root, exist_ok=True)
    (root / "volume.nii.gz").write_bytes(b"x")
    write_surface(root / "arch.vtk")

    with pytest.raises(ToolInputError) as raised:
        detect(root, tmp_path)
    message = str(raised.value)
    assert "1 CBCT scan(s)" in message
    assert "1 intraoral" in message
    assert "ALI_CBCT" in message and "ALI_IOS" in message


def test_an_empty_folder_says_what_it_was_looking_for(tmp_path):
    root = tmp_path / "in"
    os.makedirs(root, exist_ok=True)
    with pytest.raises(ToolInputError) as raised:
        detect(root, tmp_path)
    assert ".vtk" in str(raised.value) and ".stl" in str(raised.value)


def test_a_folder_of_unrelated_files_is_the_same_refusal(tmp_path):
    root = tmp_path / "in"
    os.makedirs(root, exist_ok=True)
    (root / "readme.md").write_text("nothing to see")
    (root / "labels.csv").write_text("a,b")
    with pytest.raises(ToolInputError, match="No intraoral surface"):
        detect(root, tmp_path)


def test_the_working_directory_is_never_rediscovered_as_input(tmp_path):
    """`.ali_work/` lives under the output directory, which a caller is free to
    point at the input tree. Re-reading its own intermediates would grow the
    batch on every run."""
    root = tmp_path / "in"
    write_surface(root / "arch.vtk")
    write_surface(root / WORK_DIRNAME / "leftover.vtk")

    keys = [key for _path, key in detect(root, tmp_path).scans]
    assert keys == ["arch.vtk"]
