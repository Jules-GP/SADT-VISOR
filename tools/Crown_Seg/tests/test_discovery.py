"""What counts as an input mesh, and what the tool says when nothing does.

`meshes` is one surface or a folder of them, searched recursively; the tree it
finds is what the output mirrors. Every message here reaches a caller as a 422,
so each one is asserted to NAME what was wrong rather than merely to be raised.
"""

import os

import pytest

from sadt_crownseg import pipeline
from sadt_crownseg.errors import ToolInputError

from conftest import write_stl, write_surface


# ---------------------------------------------------------------------------
# One file
# ---------------------------------------------------------------------------

def test_a_single_mesh_is_its_own_batch(tmp_path):
    mesh = write_surface(tmp_path / "cohort" / "arch.vtk")

    assert pipeline.discover_meshes(mesh) == [mesh]


def test_a_single_stl_is_accepted(tmp_path):
    """Both extensions shapeaxi's reader handles. The Slicer module counted
    .stl in its UI and then globbed for .vtk only, so they were silently never
    processed."""
    mesh = write_stl(tmp_path / "cohort" / "arch.stl", tmp_path=tmp_path)

    assert pipeline.discover_meshes(mesh) == [mesh]


@pytest.mark.parametrize("name", ["arch.VTK", "arch.Vtk", "arch.STL", "arch.Stl"])
def test_the_extension_is_matched_case_insensitively(tmp_path, name):
    """A mesh exported on Windows arrives shouting often enough."""
    path = str(tmp_path / "cohort" / name)
    if name.lower().endswith(".stl"):
        write_stl(path, tmp_path=tmp_path)
    else:
        write_surface(path)

    assert pipeline.discover_meshes(path) == [path]


def test_a_compound_name_is_read_by_its_last_extension(tmp_path):
    """`T1_01_U.seg.vtk` is a .vtk; only the tail is matched."""
    mesh = write_surface(tmp_path / "cohort" / "T1_01_U.seg.vtk")

    assert pipeline.discover_meshes(mesh) == [mesh]


def test_a_file_of_the_wrong_kind_names_itself_and_what_was_expected(tmp_path):
    """The message is the 422 body. Naming only "the input" leaves a clinician
    guessing which of forty files is the problem."""
    (tmp_path / "notes.txt").write_text("not a mesh")

    with pytest.raises(ToolInputError) as failure:
        pipeline.discover_meshes(str(tmp_path / "notes.txt"))

    assert "notes.txt" in str(failure.value)
    assert ".vtk" in str(failure.value) and ".stl" in str(failure.value)


def test_a_zip_is_not_extracted_by_the_tool(tmp_path):
    """The server unpacks archives before `run()`; the port deleted its half,
    so an archive that reaches here is simply the wrong kind of file."""
    archive = tmp_path / "cohort.zip"
    archive.write_bytes(b"PK\x03\x04")

    with pytest.raises(ToolInputError, match="cohort.zip"):
        pipeline.discover_meshes(str(archive))


def test_a_name_that_only_contains_vtk_is_not_a_mesh(tmp_path):
    """`arch.vtk.bak` ends in .bak. Matching anywhere in the name would make a
    backup file an input."""
    (tmp_path / "arch.vtk.bak").write_bytes(b"")

    with pytest.raises(ToolInputError, match="arch.vtk.bak"):
        pipeline.discover_meshes(str(tmp_path / "arch.vtk.bak"))


def test_a_path_that_does_not_exist_names_the_path(tmp_path):
    """FileNotFoundError, not ToolInputError: a path the server built and then
    could not find is the server's bug, not the caller's."""
    with pytest.raises(FileNotFoundError) as failure:
        pipeline.discover_meshes(str(tmp_path / "absent"))

    assert str(tmp_path / "absent") in str(failure.value)


# ---------------------------------------------------------------------------
# A folder
# ---------------------------------------------------------------------------

def test_discovery_is_recursive_to_any_depth(tmp_path):
    write_surface(tmp_path / "cohort" / "a" / "b" / "c" / "deep.vtk")
    write_surface(tmp_path / "cohort" / "top.vtk")

    found = pipeline.discover_meshes(str(tmp_path / "cohort"))

    assert [os.path.basename(path) for path in found] == ["deep.vtk", "top.vtk"]


def test_discovery_is_sorted(tmp_path):
    """The order decides the csv order, which decides the order shapeaxi
    reports and therefore the order of `segmented_meshes`. Filesystem order
    would make two runs on the same cohort disagree."""
    for name in ("z.vtk", "a.vtk", "m.vtk"):
        write_surface(tmp_path / "cohort" / name)

    found = pipeline.discover_meshes(str(tmp_path / "cohort"))

    assert found == sorted(found)
    assert [os.path.basename(path) for path in found] == ["a.vtk", "m.vtk", "z.vtk"]


def test_a_folder_walk_matches_the_extension_case_insensitively(tmp_path):
    """The single-file check and the folder walk are two separate tests of the
    same thing; a cohort exported on Windows arrives as ARCH.VTK, and matching
    it in only one of the two places loses the whole batch."""
    write_surface(tmp_path / "cohort" / "ARCH.VTK")
    write_stl(tmp_path / "cohort" / "OTHER.STL", tmp_path=tmp_path)

    found = pipeline.discover_meshes(str(tmp_path / "cohort"))

    assert [os.path.basename(path) for path in found] == ["ARCH.VTK", "OTHER.STL"]


def test_both_extensions_are_discovered_together(tmp_path):
    write_surface(tmp_path / "cohort" / "a.vtk")
    write_stl(tmp_path / "cohort" / "b.stl", tmp_path=tmp_path)

    found = pipeline.discover_meshes(str(tmp_path / "cohort"))

    assert [os.path.basename(path) for path in found] == ["a.vtk", "b.stl"]


def test_files_of_other_kinds_in_the_folder_are_ignored(tmp_path):
    write_surface(tmp_path / "cohort" / "arch.vtk")
    (tmp_path / "cohort" / "README.md").write_text("collected 2026")
    (tmp_path / "cohort" / "landmarks.mrk.json").write_text("{}")

    found = pipeline.discover_meshes(str(tmp_path / "cohort"))

    assert [os.path.basename(path) for path in found] == ["arch.vtk"]


def test_a_hidden_mesh_is_discovered_like_any_other(tmp_path):
    """Pinned because it is a real difference between filesystems and zip
    tools: a leading dot is not a marker here, and a `.arch.vtk` unpacked from
    an archive is a patient's scan like any other."""
    write_surface(tmp_path / "cohort" / ".arch.vtk")

    found = pipeline.discover_meshes(str(tmp_path / "cohort"))

    assert [os.path.basename(path) for path in found] == [".arch.vtk"]


def test_a_mesh_under_a_hidden_directory_is_discovered_too(tmp_path):
    write_surface(tmp_path / "cohort" / ".staging" / "arch.vtk")

    assert len(pipeline.discover_meshes(str(tmp_path / "cohort"))) == 1


def test_an_empty_folder_says_which_extensions_it_wanted(tmp_path):
    (tmp_path / "cohort").mkdir()

    with pytest.raises(ToolInputError) as failure:
        pipeline.discover_meshes(str(tmp_path / "cohort"))

    assert ".vtk" in str(failure.value) and ".stl" in str(failure.value)


def test_a_folder_of_nothing_but_other_files_is_an_empty_input(tmp_path):
    (tmp_path / "cohort").mkdir()
    (tmp_path / "cohort" / "notes.txt").write_text("x")

    with pytest.raises(ToolInputError, match="No surface mesh"):
        pipeline.discover_meshes(str(tmp_path / "cohort"))


def test_a_folder_of_only_empty_subfolders_is_an_empty_input(tmp_path):
    (tmp_path / "cohort" / "siteA").mkdir(parents=True)

    with pytest.raises(ToolInputError, match="No surface mesh"):
        pipeline.discover_meshes(str(tmp_path / "cohort"))


# ---------------------------------------------------------------------------
# The root the output tree is measured from
# ---------------------------------------------------------------------------

def test_the_input_root_of_a_folder_is_the_folder(tmp_path):
    meshes = [write_surface(tmp_path / "cohort" / "siteA" / "arch.vtk")]

    root = pipeline._input_root(str(tmp_path / "cohort"), meshes)

    assert root == str(tmp_path / "cohort")


def test_the_input_root_of_a_single_file_is_its_own_directory(tmp_path):
    """So a single mesh lands at the top of the output rather than under a
    reconstruction of its absolute path on the server."""
    mesh = write_surface(tmp_path / "cohort" / "arch.vtk")

    assert pipeline._input_root(mesh, [mesh]) == str(tmp_path / "cohort")


# ---------------------------------------------------------------------------
# Reading a mesh
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("array_name", pipeline.LABEL_ARRAY_NAMES)
def test_each_known_label_array_is_recognised_on_its_own(tmp_path, array_name):
    """Three spellings are in circulation -- shapeaxi's own output, and two
    from the Slicer modules. Any of them means the mesh is done."""
    mesh = write_surface(tmp_path / "arch.vtk", labelled=True, array_name=array_name)

    assert pipeline.is_segmented(mesh) is True


def test_an_array_by_another_name_is_not_a_label_array(tmp_path):
    """Meshes routinely carry Normals or a curvature field; neither is a tooth
    number, and treating one as such would skip the segmentation entirely."""
    mesh = write_surface(tmp_path / "arch.vtk", labelled=True, array_name="Curvature")

    assert pipeline.is_segmented(mesh) is False


def test_a_file_the_reader_cannot_parse_is_simply_not_segmented(tmp_path):
    """VTK answers a corrupt file with an empty dataset rather than an
    exception. That must read as "not labelled" -- the mesh then goes to the
    engine, which fails it individually -- and never take the batch down."""
    broken = tmp_path / "arch.vtk"
    broken.write_bytes(b"this is not a vtk file")

    assert pipeline.is_segmented(str(broken)) is False


def test_an_unknown_extension_is_never_read_at_all(tmp_path):
    """`is_segmented` is asked before anything is opened, so a file it does not
    recognise must answer without constructing a reader for it."""
    other = tmp_path / "notes.txt"
    other.write_text("x")

    assert pipeline.is_segmented(str(other)) is False


def test_deciding_whether_a_mesh_is_labelled_reads_no_value():
    """This decides whether to spend minutes of GPU time, and nothing about the
    patient. Only the array NAMES are read -- asserted on the source, because a
    single `GetTuple` added here would read patient data into the process that
    logs."""
    import inspect

    source = inspect.getsource(pipeline.is_segmented)
    assert "GetArrayName" in source
    for reader in ("GetValue", "GetTuple", "GetRange", "numpy_support"):
        assert reader not in source, reader
