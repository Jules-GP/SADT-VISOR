"""How a client lays this tool's panel out. Presentation only."""

_INPUTS = "Inputs"
_OPTIONS = "Options"
_OUTPUTS = "Outputs"

LAYOUT = {
    "files": {"section": _INPUTS, "label": "Files to transform"},
    "transforms": {"section": _INPUTS, "label": "Transforms"},

    # An INPUT, not a model. The server's convention reads any argument named
    # `reference` as a hosted bundle and files it under "Model", which put a
    # volume the caller supplies in a box next to things they cannot supply --
    # and, with `DATA/AutoMatrix/models/` empty, behind a dropdown with nothing
    # in it. deployment.toml takes it out of that convention; this puts it back
    # beside the files it belongs with.
    "reference": {"section": _INPUTS, "label": "Reference volume"},

    # "Content" said nothing about what the choice decides. It decides how the
    # voxels are resampled, and getting it wrong INVENTS labels that were never
    # segmented -- so the label names the thing being transformed, which is what
    # a clinician knows about their own files.
    "content": {"section": _OPTIONS, "label": "What these files are"},

    # The generated label ran to "Name output after transform" and was cut off
    # mid-word in the panel. What it does is add the transform's name, which is
    # how two transforms of one patient stay apart.
    "name_output_after_transform": {
        "section": _OUTPUTS, "label": "Include the transform name",
    },
    "output_suffix": {"section": _OUTPUTS, "label": "Output suffix"},
}
