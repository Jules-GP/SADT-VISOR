"""The IOS landmark vocabulary: teeth, landmark types, and the networks.

Three networks exist, and each predicts a fixed set of landmark types on a
tooth it is pointed at:

* **Occlusal** (`O`) -- the occlusal point plus the mesio- and disto-buccal
  cusps (`O`, `MB`, `DB`);
* **Cervical** (`C`) -- the cervical lingual and buccal points (`CL`, `CB`);
* **Mucogingival** (`MG`) -- one point per lower tooth, on the gingival margin
  rather than on the crown. It exists so a caller can register a mandible on
  the band around the mucogingival line without predicting those landmarks
  somewhere else first.

`R`, `RIP` and `OIP` were offered by the Slicer UI and are deliberately absent
here: no shipped model predicts them, and ticking them did nothing at all.
"""

# Teeth in Universal numbering, per jaw. Upper is 2..15 right-to-left, lower is
# 18..31 left-to-right; the weights index their label tables this way, so the
# order is part of the models' contract, not a presentation choice.
UPPER_TEETH = ["UL7", "UL6", "UL5", "UL4", "UL3", "UL2", "UL1",
               "UR1", "UR2", "UR3", "UR4", "UR5", "UR6", "UR7"]
LOWER_TEETH = ["LL7", "LL6", "LL5", "LL4", "LL3", "LL2", "LL1",
               "LR1", "LR2", "LR3", "LR4", "LR5", "LR6", "LR7"]

# Universal number of every tooth, per jaw.
UNIVERSAL_NUMBERS = {
    "Upper": {name: 15 - index for index, name in enumerate(UPPER_TEETH)},
    "Lower": {name: 18 + index for index, name in enumerate(LOWER_TEETH)},
}

# Network -> the landmark types it predicts, mapped to the channel of its
# output the type comes out on.
NETWORKS = {
    "O": {"O": 0, "MB": 1, "DB": 2},
    "C": {"CL": 0, "CB": 1},
    "MG": {"MG": 0},
}

# What the schema publishes as `choices`: display name -> network code.
NETWORK_NAMES = {"Occlusal": "O", "Cervical": "C", "Mucogingival": "MG"}
NETWORK_CODES = tuple(NETWORK_NAMES.values())
NETWORK_DISPLAY_NAMES = {code: display for display, code in NETWORK_NAMES.items()}

# The networks that only exist for one jaw. MG was trained on the mandible
# alone, so asking for it on a maxilla is not a missing model, it is a question
# with no answer.
NETWORK_JAWS = {"MG": ("Lower",)}

# Radius at which the agent's cameras orbit the tooth, per network. The
# cervical points sit lower on the crown and need a wider view than the
# occlusal ones; MG shares the occlusal radius but aims its cameras elsewhere
# (see MG_AIM_OFFSET).
CAMERA_RADIUS = {"O": 0.2, "C": 0.3, "MG": 0.2}

# {network: {universal number as str: [landmark label per channel]}}.
# `UR1O`, `UR1MB`, ... -- the tooth name followed by the type.
LABELS = {
    network: {
        str(number): [f"{tooth}{lm_type}" for lm_type in types]
        for jaw, teeth in UNIVERSAL_NUMBERS.items()
        for tooth, number in teeth.items()
    }
    for network, types in (("O", ("O", "MB", "DB")), ("C", ("CL", "CB")))
}

# ---------------------------------------------------------------------------
# Mucogingival
# ---------------------------------------------------------------------------

# Names written in the predicted MG file, assigned positionally to the 13
# trained teeth taken in arch order (universal ids 19 -> 31). Tooth 25 carries
# the midline name L0MG, so the right side is shifted by one against the tooth
# numbers: LR1MG sits on tooth 26, not 25. Tooth 18 has no MG label (excluded
# from training).
#
# Six of these names collide with the TRAINING name of a different tooth (LR1MG
# is the training name of tooth 25 and the output name of tooth 26), which is
# why the table is positional and why nothing here translates a label on its
# own.
MG_OUTPUT_NAME = (
    "LL6MG", "LL5MG", "LL4MG", "LL3MG", "LL2MG", "LL1MG", "L0MG",
    "LR1MG", "LR2MG", "LR3MG", "LR4MG", "LR5MG", "LR6MG",
)

MG_TEETH = tuple(range(19, 32))

# Where the MG landmark sits relative to the centre of its tooth, in
# unit-sphere space, in the tooth's local frame: (buccal, along-the-arch,
# vertical). Median over upstream's 155 training scans, spread 0.02-0.03 on the
# buccal and vertical axes -- a stable anatomical prior, not a per-scan fit.
#
# The cameras aim here instead of at a flat "0.2 below the tooth centre", which
# only ever matched the incisors: on the molars the landmark is ~0.15 further
# buccal, which is why it fell outside the render entirely.
MG_AIM_OFFSET = {
    19: (0.149, -0.154, -0.125),   # LL6
    20: (0.130, -0.099, -0.156),   # LL5
    21: (0.077, -0.106, -0.164),   # LL4
    22: (0.035, -0.110, -0.182),   # LL3
    23: (0.012, -0.071, -0.200),   # LL2
    24: (-0.008, -0.063, -0.202),  # LL1
    25: (-0.014, -0.057, -0.189),  # L0
    26: (-0.008, -0.059, -0.199),  # LR1
    27: (0.003, -0.074, -0.190),   # LR2
    28: (0.058, -0.073, -0.186),   # LR3
    29: (0.096, -0.082, -0.171),   # LR4
    30: (0.135, -0.124, -0.168),   # LR5
    31: (0.160, -0.107, -0.156),   # LR6
}

# MG is positional rather than "<tooth><type>", for the shift described above.
LABELS["MG"] = {
    str(number): [MG_OUTPUT_NAME[index]] for index, number in enumerate(MG_TEETH)
}


# ---------------------------------------------------------------------------
# What each landmark IS, in words
# ---------------------------------------------------------------------------

# A landmark is a CODE. `UR1MB` names no anatomy a clinician can read off it,
# and the argument's own description covers all 153 at once -- so the schema
# publishes one line per option and the panel shows it on the option itself.

# Position in its quadrant, counted from the midline outward. The universal
# numbers above are the authority; this is the same arch read as words.
_TEETH_FROM_MIDLINE = (
    "central incisor", "lateral incisor", "canine", "first premolar",
    "second premolar", "first molar", "second molar", "third molar",
)


def _tooth_name(number: int) -> str:
    """Universal number -> the tooth it names.

    Universal numbering runs 1-16 across the upper arch from the patient's
    RIGHT, then 17-32 across the lower arch from the patient's LEFT, so each
    quadrant counts toward the midline or away from it depending which one it
    is. Derived rather than tabulated, so it cannot drift from the numbers.
    """
    if 1 <= number <= 8:
        return "upper right " + _TEETH_FROM_MIDLINE[8 - number]
    if 9 <= number <= 16:
        return "upper left " + _TEETH_FROM_MIDLINE[number - 9]
    if 17 <= number <= 24:
        return "lower left " + _TEETH_FROM_MIDLINE[24 - number]
    if 25 <= number <= 32:
        return "lower right " + _TEETH_FROM_MIDLINE[number - 25]
    return "tooth {}".format(number)


# Where on the tooth the point sits. The words are this module's own header:
# Occlusal is the occlusal point plus the mesio- and disto-buccal ones,
# Cervical the cervical lingual and buccal, Mucogingival the gingival margin.
_POINT_NAMES = {
    "O": "occlusal point",
    "MB": "mesio-buccal point",
    "DB": "disto-buccal point",
    "CL": "cervical lingual point",
    "CB": "cervical buccal point",
    "MG": "gingival margin point",
}


def _describe_landmarks() -> dict:
    """{landmark: "<tooth> -- <point> (universal <n>)"}, from LABELS.

    Read out of `LABELS` and never off the label's own spelling. That is not
    fussiness: the mucogingival names are assigned POSITIONALLY and the midline
    name shifts the right side by one, so `LR1MG` sits on tooth 26 and parsing
    it as "LR1" would put it on 25 -- naming the wrong tooth in a tooltip a
    clinician is about to trust. `LABELS` is where the truth already is.
    """
    described = {}
    for network, table in LABELS.items():
        for number, labels in table.items():
            tooth = _tooth_name(int(number))
            for label in labels:
                # Longest first: "MB" and "B" would both match a name ending in
                # "MB", and the longer one is the one that means something.
                suffix = next((code for code in sorted(_POINT_NAMES, key=len, reverse=True)
                               if label.endswith(code)), "")
                point = _POINT_NAMES.get(suffix, "landmark")
                described[label] = "{} -- {} (universal {})".format(tooth, point, number)
    return described


DESCRIPTIONS = _describe_landmarks()

# Jaw a Universal tooth number belongs to.
JAW_OF_NUMBER = {
    number: jaw for jaw, teeth in UNIVERSAL_NUMBERS.items() for number in teeth.values()
}

JAWS = ("Upper", "Lower")


# ---------------------------------------------------------------------------
# The landmarks, one by one
# ---------------------------------------------------------------------------


def _landmarks_of(network: str, jaw: str) -> tuple:
    """Every landmark `network` predicts on `jaw`, in the label tables' order."""
    return tuple(
        label
        for number, names in LABELS[network].items()
        if JAW_OF_NUMBER[int(number)] == jaw
        for label in names
    )


# Tab -> the landmarks it holds. Derived from the label tables above, never
# restated: a landmark added to one appears in its tab with no edit here and no
# client release.
#
# Family x jaw rather than family alone because a family is 84 check boxes
# while an intraoral scan is one arch -- the 42 maxillary options are noise
# beside a mandible. A group with no landmarks is dropped rather than shown
# empty, which is what keeps Mucogingival, trained on the mandible alone, from
# offering an upper tab nothing could ever fill.
LANDMARK_GROUPS = {
    f"{display} {jaw}": _landmarks_of(code, jaw)
    for display, code in NETWORK_NAMES.items()
    for jaw in JAWS
    if _landmarks_of(code, jaw)
}

# Every landmark this tool can place, in the order the tabs present them.
LANDMARKS = tuple(label for labels in LANDMARK_GROUPS.values() for label in labels)

# Landmark -> the network that predicts it. The three tables share no name -- a
# test pins that -- so one label names exactly one forward pass; a collision
# would make a selection ambiguous rather than merely wrong.
LANDMARK_NETWORK = {
    label: network
    for network, table in LABELS.items()
    for names in table.values()
    for label in names
}


def network_codes(selection) -> tuple:
    """Turn what `run()` received for `ios_networks` into network codes.

    Display names ("Occlusal") are what the schema publishes and what a client
    sends; the codes ("O") are accepted too. None means the argument was
    omitted and both networks are wanted. An unknown name raises, for the same
    reason `cbct.catalog.region_codes` refuses one.
    """
    if selection is None:
        return NETWORK_CODES
    codes = []
    for name in selection:
        if name in NETWORK_NAMES:
            codes.append(NETWORK_NAMES[name])
        elif name in NETWORK_CODES:
            codes.append(name)
        else:
            raise ValueError(
                f"Unknown IOS landmark family {name!r}. Known: {', '.join(NETWORK_NAMES)}."
            )
    return tuple(code for code in NETWORK_CODES if code in set(codes))


def resolve_landmarks(selection) -> tuple:
    """`(recognised, unknown)` for what `run()` received for `landmarks`.

    Empty is the ordinary case and means "not specified": the networks decide,
    which is what every request written before this argument existed relies on.

    Unlike ALI_CBCT's counterpart, a name this catalog does not know CANNOT be
    honoured here: a network emits a fixed set of channels and there is no
    bundle-provided extra to fall back on. Unknown names are returned rather
    than dropped, so the run report can say what the selection did not buy.

    Recognised names come back in declaration order, not the caller's, so the
    report reads the same way however the request was assembled.
    """
    if not selection:
        return (), ()
    wanted = set(selection)
    recognised = tuple(label for label in LANDMARKS if label in wanted)
    return recognised, tuple(sorted(wanted - set(recognised)))


def networks_for(labels) -> tuple:
    """The networks that must run to produce these landmarks.

    This is what makes naming landmarks cheaper than naming the family they
    belong to: asking for the 13 mucogingival points runs the MG pass alone,
    and never the occlusal and cervical passes the default would have run over
    every mesh.
    """
    wanted = {LANDMARK_NETWORK[label] for label in labels if label in LANDMARK_NETWORK}
    return tuple(code for code in NETWORK_CODES if code in wanted)
