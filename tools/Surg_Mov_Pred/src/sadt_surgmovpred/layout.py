"""How a client lays Surg_Mov_Pred's panel out.

Nothing here changes what `run()` computes.
"""

LAYOUT = {
    # Not shown. One set of models is published, the tool loads every package it
    # finds under it, and since `model` became optional it finds them itself --
    # so the field asked a clinician to confirm the only answer there is, in a
    # section of its own, above the one input that matters.
    #
    # Hidden rather than removed: the argument stays on the API, so a caller
    # comparing two sets of models can still pin one.
    "model": {"hidden": True},

    # "Measurements" alone does not say measurements of WHAT, and this is the
    # only field left in the panel. Not "cephalometric measurements" either,
    # which is only half of what the table holds: read off the shipped models'
    # own feature lists, each of the 112 packages asks for the same 145 columns
    # -- 82 landmark-to-plane distances (Frankfort, midsagittal, coronal), a set
    # of classical angles and ratios, 28 of them suffixed `_T0` for the
    # pre-operative state, and 25 suffixed `_Total` for the total movement.
    # That last family is the intended correction, not a measurement of the
    # patient, so a label naming only measurements hides half the input.
    "measurements": {"label": "Measurements and plan"},
}
