"""How a client lays this tool's panel out. Presentation only."""

_INPUTS = "Inputs"
_OUTPUTS = "Outputs"
_ADVANCED = "Advanced"

LAYOUT = {
    "files": {"section": _INPUTS, "label": "Files to transform"},
    "transforms": {"section": _INPUTS, "label": "Transforms"},

    # An INPUT, not a model. The server's convention reads any argument named
    # `reference` as a hosted bundle and files it under "Model", which put a
    # volume the caller supplies in a box next to things they cannot supply --
    # and, with `DATA/AutoMatrix/models/` empty, behind a dropdown with nothing
    # in it. deployment.toml takes it out of that convention.
    #
    # It lands in "Advanced" rather than beside the files because nothing needs
    # it: with no reference each image keeps its own grid and only its origin
    # moves, which is what a caller transforming their own scans wants. Offered
    # among the inputs it reads as a field to fill in, and a wrong volume here
    # resamples every output onto a grid that is not the patient's.
    # Hidden, not offered. It defines the output grid, and with none each image
    # keeps its own -- which is what every caller here wants, VFACE included
    # (it passes "None" at all eight of its call sites). Offered, it read as a
    # field to fill in, and a wrong volume resamples every output onto a grid
    # that is not the patient's. Still an argument, so an API caller who has a
    # reason can pass one.
    "reference": {"section": _ADVANCED, "label": "Reference volume", "hidden": True},

    # "Content" said nothing about what the choice decides. It decides how the
    # voxels are resampled, and getting it wrong INVENTS labels that were never
    # segmented -- so the label names the thing being transformed, which is what
    # a clinician knows about their own files.
    # Folded away, and defaulting to "Automatic". It decides the interpolator,
    # and reading that off the file is both more reliable than asking and finer
    # grained: the answer is per FILE, where the argument was per run -- which is
    # what a folder holding scans AND segmentations always needed. Kept, rather
    # than removed, so a caller with an unusual volume can still overrule it.
    # Hidden too, now that "Automatic" reads it off each file. It asked a
    # clinician to answer a question about interpolation -- and to answer it once
    # for a whole folder, where the truth is per file. Kept as an argument so a
    # caller with an unusual volume can still overrule the data.
    "content": {"section": _ADVANCED, "label": "How to resample", "hidden": True},

    # The generated label ran to "Name output after transform" and was cut off
    # mid-word in the panel. What it does is add the transform's name, which is
    # how two transforms of one patient stay apart.
    "name_output_after_transform": {
        "section": _OUTPUTS, "label": "Include the transform name",
    },
    "output_suffix": {"section": _OUTPUTS, "label": "Output suffix"},

    # Only ever needed with ONE transform, and then it is the whole intent: a
    # mirror matrix belongs to no patient in particular. Left off, a cohort and
    # a single transform that names nobody is refused rather than guessed at.
    "same_transform_for_every_patient": {
        "section": _INPUTS, "label": "Same transform for every patient",
    },
}
