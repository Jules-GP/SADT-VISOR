"""How a client lays Crown_Seg's panel out.

Nothing here changes what `run()` computes.
"""

LAYOUT = {
    # Not shown. One checkpoint is published, and since `model` became optional
    # the tool finds it itself -- `find_checkpoint` already walks a folder and
    # refuses to choose between two vintages, so the field asked a clinician to
    # confirm the only answer there is.
    #
    # Hidden rather than removed: the argument stays on the API, so a caller
    # comparing two vintages can still pin one, and the four supervised callers
    # can still pass the path they were handed.
    "model": {"hidden": True},

    # Not shown either, and for a different reason: this is a CONTRACT, not a
    # choice. It names the VTK point-data array the tooth numbers are written
    # into, and every tool that consumes this mesh looks for one of
    # `Universal_ID`, `PredictedID`, `UniversalID` -- ALI's IOS landmarks, ASO,
    # AREG and FlexReg all do. A clinician typing anything else gets a mesh
    # that is correctly labelled and that no downstream tool can read, which is
    # a failure with no symptom until the next tool in the chain.
    #
    # It also no longer needs answering: left empty it follows `numbering`, so
    # the name cannot disagree with the integers underneath it.
    #
    # `numbering` is the choice that IS clinical, and it stays: it changes the
    # integers written into that array.
    "array_name": {"hidden": True},

    # What the two numbering systems are, since the bare names do not say.
    "numbering": {
        "label": "Tooth numbering",
    },
}
