"""How a client lays CLIC's panel out.

Nothing here changes what `run()` computes.
"""

LAYOUT = {
    # Not shown. There is ONE published checkpoint, and since `model` became
    # optional the tool finds it itself -- so the field asked a clinician to
    # confirm the only answer there is, in a section of its own, above the
    # threshold that actually matters.
    #
    # Hidden rather than removed: the argument still exists on the API, so a
    # caller comparing two model vintages can still pin one, and a supervised
    # call can still pass a path. What it costs is that a deployment staging a
    # SECOND checkpoint makes the tool refuse (it will not guess which weights
    # ran) with no way to choose from the panel -- which is the right trade
    # while the manifest ships one, and one line to undo if that changes.
    "model": {"hidden": True},
}
