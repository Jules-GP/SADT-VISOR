"""What `discover_scans` walks, what it refuses, and in what order.

Upstream's collector returned SUBDIRECTORIES whenever any of them held a scan,
looked one level down, and advertised three extensions nibabel cannot read.
Every test here pins one half of that.
"""

import os

import pytest

from sadt_clic import pipeline


def test_a_single_file_is_returned_as_itself(tmp_path, make_scan):
    """A scan named directly is a batch of one, not an empty walk."""
    scan = make_scan(tmp_path / "patient.nii.gz")

    assert pipeline.discover_scans(str(scan)) == [str(scan)]


def test_a_single_file_of_the_wrong_kind_is_not_a_batch_of_one(tmp_path):
    """`.nrrd` named directly must not reach `nib.load`, which cannot read it."""
    volume = tmp_path / "patient.nrrd"
    volume.write_bytes(b"NRRD0004")

    assert pipeline.discover_scans(str(volume)) == []


def test_an_empty_folder_finds_nothing(tmp_path):
    (tmp_path / "empty").mkdir()

    assert pipeline.discover_scans(str(tmp_path / "empty")) == []


def test_a_folder_of_unreadable_extensions_only_finds_nothing(tmp_path):
    """The extensions upstream advertised and could not read are not accepted
    here, so a folder of them is empty rather than a batch that fails one scan
    at a time inside the loader."""
    folder = tmp_path / "in"
    folder.mkdir()
    for name in ("a.nrrd", "b.mha", "c.mhd", "d.dcm", "notes.txt"):
        (folder / name).write_bytes(b"x")

    assert pipeline.discover_scans(str(folder)) == []


def test_a_missing_path_is_empty_rather_than_an_error(tmp_path):
    """`run` turns an empty result into a 422 naming what was looked for; an
    OSError out of the walk would be a 500 instead."""
    assert pipeline.discover_scans(str(tmp_path / "nowhere")) == []


def test_scans_come_back_sorted(tmp_path, make_scan):
    """Written out of order on purpose: a batch's order must not depend on the
    order the filesystem happens to hand names back."""
    for name in ("zeta.nii.gz", "alpha.nii.gz", "mu.nii.gz"):
        make_scan(tmp_path / name)

    found = pipeline.discover_scans(str(tmp_path))

    assert found == sorted(found)
    assert [os.path.basename(f) for f in found] == [
        "alpha.nii.gz", "mu.nii.gz", "zeta.nii.gz",
    ]


def test_sorting_holds_across_subfolders(tmp_path, make_scan):
    """The sort is over full paths, so two folders interleave predictably."""
    make_scan(tmp_path / "b" / "scan.nii.gz")
    make_scan(tmp_path / "a" / "scan.nii.gz")

    found = pipeline.discover_scans(str(tmp_path))

    assert found == [
        str(tmp_path / "a" / "scan.nii.gz"),
        str(tmp_path / "b" / "scan.nii.gz"),
    ]


def test_hidden_files_are_skipped_by_a_walk(tmp_path, make_scan):
    """`._patient.nii.gz` is the resource fork a macOS zip carries beside the
    real file. Loading it is a per-scan failure in every batch that came from a
    Mac."""
    make_scan(tmp_path / "patient.nii.gz")
    make_scan(tmp_path / "._patient.nii.gz")
    make_scan(tmp_path / ".hidden.nii.gz")

    found = pipeline.discover_scans(str(tmp_path))

    assert [os.path.basename(f) for f in found] == ["patient.nii.gz"]


def test_a_hidden_file_named_directly_is_still_read(tmp_path, make_scan):
    """The skip is a heuristic for a walk. Naming a file is explicit, and an
    explicit request must not be silently dropped."""
    scan = make_scan(tmp_path / ".patient.nii.gz")

    assert pipeline.discover_scans(str(scan)) == [str(scan)]


def test_both_nii_and_nii_gz_are_found(tmp_path, make_scan):
    make_scan(tmp_path / "plain.nii")
    make_scan(tmp_path / "zipped.nii.gz")

    found = pipeline.discover_scans(str(tmp_path))

    assert [os.path.basename(f) for f in found] == ["plain.nii", "zipped.nii.gz"]


def test_the_extension_check_is_case_insensitive_in_a_walk(tmp_path):
    """A scan exported from Windows arrives as `.NII.GZ` and is the same file.

    Written as plain bytes rather than through nibabel, which normalises the
    case of the extension it is handed and would decide the test.
    """
    for name in ("A.NII", "B.NII.GZ", "C.Nii.Gz", "D.nII"):
        (tmp_path / name).write_bytes(b"")

    found = pipeline.discover_scans(str(tmp_path))

    assert [os.path.basename(f) for f in found] == ["A.NII", "B.NII.GZ", "C.Nii.Gz", "D.nII"]


@pytest.mark.parametrize("name", [
    "scan.nii.gz.txt",     # a note about the scan, not the scan
    "scan.nifti",
    "scan.gz",             # gzipped something else
    "scan.nii.zip",
    "nii.gz",              # the extension with no name in front of it
    "readme_nii_gz.md",
])
def test_a_misleading_name_is_not_a_scan(name):
    """The check is on the extension, not on the string appearing somewhere."""
    assert pipeline.is_scan_file(name) is False


def test_a_symlink_to_a_scan_is_followed(tmp_path, make_scan):
    """A staged batch is often symlinks into a read-only store."""
    real = make_scan(tmp_path / "store" / "patient.nii.gz")
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "link.nii.gz").symlink_to(real)

    found = pipeline.discover_scans(str(batch))

    assert [os.path.basename(f) for f in found] == ["link.nii.gz"]


def test_a_symlinked_directory_is_not_descended(tmp_path, make_scan):
    """`os.walk` does not follow directory links, and that is the behaviour to
    keep: a link out of the request's own folder would read scans the caller
    never uploaded, and a link back into it loops forever."""
    make_scan(tmp_path / "elsewhere" / "other_patient.nii.gz")
    batch = tmp_path / "batch"
    batch.mkdir()
    make_scan(batch / "mine.nii.gz")
    (batch / "escape").symlink_to(tmp_path / "elsewhere", target_is_directory=True)

    found = pipeline.discover_scans(str(batch))

    assert [os.path.basename(f) for f in found] == ["mine.nii.gz"]


def test_a_deeply_nested_tree_is_reached(tmp_path, make_scan):
    """Upstream looked one level down, so a cohort exported as
    site/patient/timepoint/ returned nothing."""
    deep = tmp_path / "site" / "patient" / "T1" / "recon" / "axial"
    make_scan(deep / "scan.nii.gz")

    found = pipeline.discover_scans(str(tmp_path))

    assert found == [str(deep / "scan.nii.gz")]


def test_a_directory_named_like_a_scan_is_never_returned(tmp_path, make_scan):
    """Upstream returned the directory and the runner called `nib.load` on it.
    A directory called `patient.nii.gz` is the case that survives every other
    guard."""
    folder = tmp_path / "in" / "patient.nii.gz"
    make_scan(folder / "real.nii.gz")

    found = pipeline.discover_scans(str(tmp_path / "in"))

    assert found == [str(folder / "real.nii.gz")]
    assert all(os.path.isfile(f) for f in found)


def test_a_directory_named_like_a_scan_passed_directly_is_walked(tmp_path, make_scan):
    """Naming it does not make it a file either: it is walked, and what comes
    back is the scans inside it."""
    folder = tmp_path / "patient.nii.gz"
    make_scan(folder / "inner.nii.gz")

    assert pipeline.discover_scans(str(folder)) == [str(folder / "inner.nii.gz")]


def test_every_result_is_an_existing_file(tmp_path, make_scan):
    """The one property the loader depends on, asserted over a mixed tree."""
    make_scan(tmp_path / "a.nii.gz")
    make_scan(tmp_path / "sub" / "b.nii")
    (tmp_path / "sub" / "c.nrrd").write_bytes(b"x")
    (tmp_path / "dir.nii").mkdir()

    found = pipeline.discover_scans(str(tmp_path))

    assert len(found) == 2
    assert all(os.path.isfile(f) for f in found)
