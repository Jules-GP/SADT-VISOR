"""Which file extensions are a volume, a surface, a set of landmarks.

**Shared rather than copied, and that is the exception this package is.**
`CONTRIBUTING.md` says small helpers are copied between tools on purpose --
`iter_scans` and `errors.py` stay duplicated, deliberately. What is shared is
what the same document calls a contract with the outside world, and it names
this table as one: ALI's two engines disagreed about `.stl` before the port,
so the UI accepted files the CLI silently ignored, and the fix was to put the
table in `tools/ALI/common/`. This is that fix, at the width the problem
actually has -- the disagreement was never between ALI's two halves alone.

Measured on 2026-09-24, before this existed: four byte-identical copies of the
scan vocabulary (AMASSS, ASO, Batch_Dental_Seg, AREG), and FIVE different
values for `SURFACE_EXTENSIONS` under one name, from `(".vtk",)` to
`(".vtk", ".vtp", ".stl", ".obj", ".off")`. Nothing in the code said which
difference was a decision and which was an oversight.

**Stdlib only, and no dependency, ever.** This installs into environments whose
pins are deliberately incompatible; anything imported here would have to be
satisfiable by all of them at once. The budget is `os`.

**A tool may read LESS than a format.** The table says what the format IS, not
what a tool handles, and advertising more than you read is the original bug:

    from sadt_formats import SURFACE
    SURFACE_EXTENSIONS = (".vtk",)   # this tool reads only this one

That narrowing is a fact about the tool and belongs beside it. What must not
be written beside it again is the list itself.
"""

import os

# A 3D image. Ordered longest-first, which `split_extension` depends on:
# `.nii.gz` has to be recognised before `.nii` or the stem keeps a `.nii`.
VOLUME = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")

# A 3D surface mesh. Every extension here is one that the FORMAT covers -- a
# tool declaring it must be able to read it, or narrow (see the module
# docstring).
SURFACE = (".vtk", ".vtp", ".stl", ".obj", ".off")

# Slicer markups: landmarks, and anything else placed by hand. `.json` is in
# because that is what the pre-port CLIs wrote for identical content, and a
# reader who opens a folder from 2024 still has those.
MARKUPS = (".mrk.json", ".json")

# A spatial transform, as ITK and Slicer write them.
TRANSFORM = (".tfm", ".mat", ".h5", ".hdf5", ".txt")

# Free-text clinical material.
NOTES = (".txt", ".pdf", ".docx")

# Tabular measurements.
TABLE = (".csv", ".xlsx", ".ods")

# The compressed spelling ITK can WRITE for each volume extension. NIfTI and
# GIPL take an external .gz; NRRD compresses inside the file and ITK has no
# ".nrrd.gz" writer at all, so that spelling maps back down to ".nrrd".
_COMPRESSED = {".nii": ".nii.gz", ".gipl": ".gipl.gz", ".nrrd.gz": ".nrrd"}


def split_extension(filename: str, extensions=VOLUME) -> tuple:
    """`('scan.nii.gz')` -> `('scan', '.nii.gz')`, compound extensions kept.

    Not `os.path.splitext`, which answers `('scan.nii', '.gz')` and leaves a
    stem no patient is called. Falls back to it for anything unrecognised, so
    a name this table says nothing about still splits somewhere sensible.
    """
    lower = filename.lower()
    for extension in extensions:
        if lower.endswith(extension):
            return filename[: -len(extension)], filename[-len(extension):]
    return os.path.splitext(filename)


def compressed_extension(extension: str) -> str:
    """The compressed spelling ITK can write for a volume extension."""
    return _COMPRESSED.get(extension.lower(), extension)


def has_extension(filename: str, extensions) -> bool:
    """Whether `filename` ends in one of `extensions`, case-insensitively.

    Written once because every tool walks a folder and every tool got this
    subtly different: `.endswith` on a bare name, on a lowered name, on a
    tuple, or a `splitext` that cannot see `.nii.gz` at all.
    """
    lower = filename.lower()
    return any(lower.endswith(extension) for extension in extensions)
