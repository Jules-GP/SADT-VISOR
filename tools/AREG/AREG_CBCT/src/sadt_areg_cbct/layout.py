"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts — `describe.py` merges these hints
into the published schema and refuses any that name an argument the signature
does not take. Delete this file and the tool still works; the panel gets worse.

No `modality` condition anywhere, and that is the split showing through. The
merged AREG carried one on almost every field, because two modalities and three
automation modes shared a single schema and most arguments applied to exactly
one combination. There is no other modality here now, so what remains are the
conditions that are really about the MODE.
"""

from sadt_areg_common import catalogs

_INPUTS = "Inputs"
_REGISTRATION = "Registration"
_OUTPUTS = "Outputs"

# Each mirrors a check in dispatch.py. An argument the chosen mode never reads
# is not merely noise: shown as optional beside the ones that matter, it reads
# as something the user chose not to fill, and the refusal then arrives at the
# end of a run instead of before it.
_SEGMENTED = {  # the modes that produce their own masks
    "automation": [catalogs.AUTOMATION_FULLY, catalogs.AUTOMATION_ORIENTED]
}
_ORIENTED = {"automation": catalogs.AUTOMATION_ORIENTED}
_SEMI = {"automation": catalogs.AUTOMATION_SEMI}

LAYOUT = {
    "t1": {"section": _INPUTS, "label": "T1 (baseline)"},
    "t2": {"section": _INPUTS, "label": "T2 (follow-up)"},
    "automation": {"section": _INPUTS, "label": "Mode"},
    "dicom_input": {"section": _INPUTS, "label": "Input is DICOM"},

    # The one argument a clinician must actually think about: register on what
    # has NOT changed between the two timepoints.
    "regions": {
        "section": _REGISTRATION,
        "label": "Register on",
        "ui": "inline",
    },
    "t1_masks": {"section": _REGISTRATION, "label": "T1 masks", "visible_when": _SEMI},
    "segmentation_model": {
        "section": _REGISTRATION, "label": "Segmentation model", "visible_when": _SEGMENTED,
    },
    "segmentation_label": {
        "section": _REGISTRATION, "label": "Mask label value", "visible_when": _SEGMENTED,
    },
    "reference": {
        "section": _REGISTRATION, "label": "Orientation reference", "visible_when": _ORIENTED,
    },
    "landmark_model": {
        "section": _REGISTRATION, "label": "Landmark model bundle", "visible_when": _ORIENTED,
    },

    "output_suffix": {"section": _OUTPUTS, "label": "Output suffix"},
}
