"""The vocabulary, and the two things it exists to stop.

One list, shared: a tool that reads less says so beside itself, and nobody
writes the list twice. Both halves are asserted here, because the failure
this package prevents is silent -- a UI offering a format the tool ignores.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sadt_formats as formats


class TestSplitting:
    def test_a_compound_extension_survives_as_one_unit(self):
        """`os.path.splitext` answers ('scan.nii', '.gz'), and no patient is
        called `scan.nii`. This is the whole reason the helper exists."""
        assert formats.split_extension("Pat_0002.nii.gz") == ("Pat_0002", ".nii.gz")
        assert formats.split_extension("Pat_0002.nrrd.gz") == ("Pat_0002", ".nrrd.gz")

    def test_the_longest_match_wins(self):
        """`.nii` comes after `.nii.gz` in the table, and reversing them
        leaves a stem ending in `.nii`."""
        assert formats.split_extension("a.nii") == ("a", ".nii")
        assert formats.split_extension("a.nii.gz")[1] == ".nii.gz"

    def test_another_vocabulary_can_be_asked_for(self):
        assert formats.split_extension("arch.vtk", formats.SURFACE) == ("arch", ".vtk")

    def test_something_the_table_says_nothing_about_still_splits(self):
        """Falling back rather than raising: a name this package has never
        heard of is not an error, it is a name."""
        assert formats.split_extension("notes.docx") == ("notes", ".docx")
        assert formats.split_extension("no_extension") == ("no_extension", "")

    def test_the_case_of_the_name_does_not_decide(self):
        assert formats.split_extension("SCAN.NII.GZ") == ("SCAN", ".NII.GZ")
        assert formats.has_extension("SCAN.NII.GZ", formats.VOLUME)


class TestWriting:
    def test_nrrd_maps_down_rather_than_up(self):
        """ITK has no `.nrrd.gz` writer at all: NRRD compresses inside the
        file. Asking for one is how a run ends in an unwritable path."""
        assert formats.compressed_extension(".nrrd.gz") == ".nrrd"
        assert formats.compressed_extension(".nii") == ".nii.gz"
        assert formats.compressed_extension(".gipl") == ".gipl.gz"

    def test_an_extension_with_no_compressed_spelling_is_returned_as_it_is(self):
        assert formats.compressed_extension(".vtk") == ".vtk"


class TestTheContract:
    def test_this_package_imports_nothing_but_the_standard_library(self):
        """It installs into environments whose pins are deliberately
        incompatible. A dependency here would have to be satisfiable by all
        of them at once, which is the constraint the split removed."""
        source = (Path(formats.__file__)).read_text(encoding="utf-8")
        imports = [line.strip() for line in source.splitlines()
                   if line.startswith(("import ", "from "))]
        assert imports == ["import os"], imports

    def test_a_tool_narrows_by_declaring_less_not_by_editing_this(self):
        """The table says what the FORMAT is. What a tool handles is a fact
        about the tool -- and advertising more than you read is the bug this
        package was created from."""
        reads_only_vtk = (".vtk",)
        assert set(reads_only_vtk) <= set(formats.SURFACE)
        assert formats.has_extension("a.stl", formats.SURFACE)
        assert not formats.has_extension("a.stl", reads_only_vtk)
