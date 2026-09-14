"""How a client lays GreedyReg's panel out.

Nothing here changes what `run()` computes.
"""

_ADVANCED = "Advanced"

LAYOUT = {
    # The two optional folders, together, out of the way of the two that are
    # not optional. Both sat in "Inputs" beside T1 and T2, where four pickers
    # in a row read as four things to fill in. The section itself stays right
    # under those two -- they are inputs, and a collapsed bar for them belongs
    # where the inputs are, not at the foot of the panel.
    #
    # `masks` is the one to reconsider if this ever feels wrong: restricting
    # the metric to a stable structure is how a growing patient is registered
    # correctly, so it is an option with real clinical weight sitting behind a
    # collapsed bar. It is still optional and unconditional, which is what puts
    # it here.
    "masks": {"section": _ADVANCED},

    # Sat beside the two timepoint folders too. It is not a fourth input: it is
    # the escape hatch for two scans so far apart that the search cannot find
    # its way from identity, and it needs one `.mat` per patient that nothing
    # in this panel produces.
    "initial_transforms": {"section": _ADVANCED},
}
