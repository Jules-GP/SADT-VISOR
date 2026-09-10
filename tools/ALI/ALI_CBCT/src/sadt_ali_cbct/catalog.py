"""The CBCT landmark vocabulary: which landmarks exist, and how they group.

This is the single source of truth the schema publishes through `choices` and
the engine validates against. The original had two copies that disagreed on the
impacted-canine spelling (`UR3OI` vs `UR3OIP`), and the cost was a lost patient
rather than a mislabelled point: an unguarded `LABEL_GROUPS[...]` in the save
loop raised a KeyError caught far above, so nothing at all was written for that
scan. Both spellings are accepted here (see `_ALIASES`) and `group_of()` never
raises.
"""

# Anatomical region -> the landmarks it contains. Region codes are the ones
# the original wrote into its output file names (CB/U/L/CI); the display names
# beside them are what the schema publishes and the client renders.
GROUP_LABELS = {
    "CB": [
        "Ba", "S", "N", "RPo", "LPo", "RFZyg", "LFZyg", "C2", "C3", "C4",
    ],
    "U": [
        "RInfOr", "LInfOr", "LMZyg", "RPF", "LPF", "PNS", "ANS", "A", "UR3O",
        "UR1O", "UL3O", "UR6DB", "UR6MB", "UL6MB", "UL6DB", "IF", "ROr", "LOr",
        "RMZyg", "RNC", "LNC", "UR7O", "UR5O", "UR4O", "UR2O", "UL1O", "UL2O",
        "UL4O", "UL5O", "UL7O", "UL7R", "UL5R", "UL4R", "UL2R", "UL1R", "UR2R",
        "UR4R", "UR5R", "UR7R", "UR6MP", "UL6MP", "UL6R", "UR6R", "UR6O",
        "UL6O", "UL3R", "UR3R", "UR1R",
    ],
    "L": [
        "RCo", "RGo", "Me", "Gn", "Pog", "PogL", "B", "LGo", "LCo", "LR1O",
        "LL6MB", "LL6DB", "LR6MB", "LR6DB", "LAF", "LAE", "RAF", "RAE", "LMCo",
        "LLCo", "RMCo", "RLCo", "RMeF", "LMeF", "RSig", "RPRa", "RARa", "LSig",
        "LARa", "LPRa", "LR7R", "LR5R", "LR4R", "LR3R", "LL3R", "LL4R", "LL5R",
        "LL7R", "LL7O", "LL5O", "LL4O", "LL3O", "LL2O", "LL1O", "LR2O", "LR3O",
        "LR4O", "LR5O", "LR7O", "LL6R", "LR6R", "LL6O", "LR6O", "LR1R", "LL1R",
        "LL2R", "LR2R",
    ],
    "CI": ["UR3OIP", "UL3OIP", "UR3RIP", "UL3RIP"],
}

# What the schema publishes as `choices`: display name -> region code. Order is
# the order the client renders the check boxes in.
REGION_NAMES = {
    "Cranial base": "CB",
    "Upper": "U",
    "Lower": "L",
    "Impacted canine": "CI",
}

REGION_CODES = tuple(REGION_NAMES.values())

# The UI spelling of the impacted-canine landmarks, mapped to the spelling the
# weights are packaged under. Accepted, never emitted.
_ALIASES = {
    "UR3OI": "UR3OIP",
    "UL3OI": "UL3OIP",
    "UR3RI": "UR3RIP",
    "UL3RI": "UL3RIP",
}

LABEL_GROUPS = {
    label: code for code, labels in GROUP_LABELS.items() for label in labels
}
LABEL_GROUPS.update({alias: LABEL_GROUPS[canonical] for alias, canonical in _ALIASES.items()})

LABELS = tuple(label for labels in GROUP_LABELS.values() for label in labels)

# The two spacings the agent walks, coarse first. The scale key is the spacing
# with its decimal point replaced, because that is what the shipped weight
# folders are named (`<landmark>/1/` and `<landmark>/0-3/`).
SCALE_SPACINGS = (1.0, 0.3)

# Region code -> a name a human reads. Used for the run report only.
REGION_DISPLAY_NAMES = {code: display for display, code in REGION_NAMES.items()}

UNGROUPED = "Other"


def scale_key(spacing: float) -> str:
    """1.0 -> '1', 0.3 -> '0-3'. The name of the weight folder for that scale."""
    text = f"{spacing:g}"
    return text.replace(".", "-")


SCALE_KEYS = tuple(scale_key(spacing) for spacing in SCALE_SPACINGS)


def canonical(label: str) -> str:
    """The spelling the vocabulary uses for `label`, aliases resolved."""
    return _ALIASES.get(label, label)


def group_of(label: str) -> str:
    """The region a landmark belongs to, or "Other" for an unknown name. Never
    raises: the unguarded lookup this replaces cost a patient every one of
    their landmarks when a bundle carried an unfamiliar name."""
    return LABEL_GROUPS.get(label, UNGROUPED)


def region_codes(selection) -> tuple:
    """Turn what `run()` received for `regions` into region codes.

    The published options are the display names ("Cranial base"), because that
    is what a client renders; the codes ("CB") are accepted too, so a caller
    driving this package directly need not know the display spellings. None
    means the argument was omitted entirely and every region is wanted.

    An unknown name raises rather than being dropped: `Literal` is published,
    not enforced -- the runner calls `run(**params)` from a JSON object -- so a
    stale client sending a region that no longer exists must be told, not
    silently given a narrower run than it asked for.
    """
    if selection is None:
        return REGION_CODES
    codes = []
    for name in selection:
        if name in REGION_NAMES:
            codes.append(REGION_NAMES[name])
        elif name in REGION_CODES:
            codes.append(name)
        else:
            raise ValueError(
                f"Unknown CBCT region {name!r}. Known: {', '.join(REGION_NAMES)}."
            )
    # Declaration order, not the caller's, so the run report reads the same way
    # however the request was assembled.
    return tuple(code for code in REGION_CODES if code in set(codes))


def landmark_names(selection) -> tuple:
    """Turn what `run()` received for `landmarks` into canonical label names.

    Empty is the normal case and means "not specified": `regions` decides.
    Aliases are resolved here (`UR3OI` -> `UR3OIP`), the weights only ever using
    the canonical spelling.

    Unlike `region_codes`, a name this catalog does not know is NOT refused: the
    engine runs any explicitly named landmark the bundle has weights for, which
    is what lets a bundle ship a point this vocabulary has not heard of. It is
    reported as `landmarks_ungrouped` in the run report.
    """
    if selection is None:
        return ()
    chosen = list(selection)
    # Declaration order, not the caller's: it is the order the report and the
    # per-group output files already use. Anything outside the catalog keeps
    # the caller's order and follows, rather than being dropped.
    wanted = {canonical(name) for name in chosen}
    known = tuple(label for label in LABELS if label in wanted)
    return known + tuple(sorted(wanted - set(known)))


def landmarks_in(codes) -> tuple:
    """Every known landmark belonging to one of these region codes."""
    wanted = set(codes)
    return tuple(
        label for code, labels in GROUP_LABELS.items() if code in wanted for label in labels
    )


# ---------------------------------------------------------------------------
# What each landmark IS, in words
# ---------------------------------------------------------------------------

# A landmark is a CODE. `UR6MB` and `Ba` name no anatomy a clinician can read
# off them, and the argument's own description covers all 119 at once -- so the
# schema publishes one line per option and the panel shows it on the option.
#
# Sourced, never guessed. Everything below comes from Gillot et al. 2023,
# "Automatic landmark identification in cone-beam computed tomography",
# Orthod Craniofac Res (PMC10440369), Table 1 -- written here in this file's
# own words rather than copied. Landmarks the paper's Table 1 does not reach
# are DELIBERATELY ABSENT: an option with no line shows no tooltip, which is
# what it had before, while a wrong line is a point placed wrongly.

# Counted from the midline outward, which is what the digit in a name gives.
# Confirmed twice: ALI_IOS's `UNIVERSAL_NUMBERS` maps `UR1` to universal 8,
# `UR3` to 6 and `UR6` to 3 (central incisor, canine, first molar in the ADA
# system), and Table 1 names `UR6` the maxillary right first molar, `UR3` the
# maxillary right canine and `UR1` the maxillary right incisor.
_FROM_MIDLINE = (
    "central incisor", "lateral incisor", "canine", "first premolar",
    "second premolar", "first molar", "second molar", "third molar",
)

_QUADRANTS = {
    "UR": "upper right", "UL": "upper left",
    "LR": "lower right", "LL": "lower left",
}

# Where on the tooth the point sits, from Table 1. `MP`, `OIP` and `RIP` are
# absent on purpose: the paper's Table 1 does not reach them, and this file
# does not invent anatomy.
# Two of them read differently along the arch, and Table 1 spells both cases
# out -- but they do not change at the same tooth. `O` is an incisal edge on an
# incisor and a cusp tip from the canine back (`UR1O` vs `UR3O`); `R` is a root
# canal on anything anterior and a pulp chamber floor on a molar (`UR3R` vs
# `UR6R`). Each suffix therefore carries its OWN boundary rather than sharing
# one, which is how a tooltip on a canine stops talking about incisal edges.
#
# (position at which the second reading starts, before it, from it on)
_POINT_NAMES = {
    "O": (3, "incisal edge, at its midpoint", "cusp tip"),
    "R": (6, "root canal, at the level of the CEJ", "pulp chamber floor, at its centre"),
    "MB": (0, "mesio-buccal cusp", "mesio-buccal cusp"),
    "DB": (0, "disto-buccal cusp", "disto-buccal cusp"),
}


# The named craniofacial points Table 1 defines. The other 46 this catalog
# offers are listed in the paper's Supplementary Table 1, which names them
# without defining them -- so they are left for a clinician to fill in.
_NAMED_POINTS = {
    "Ba": "Basion -- the anterior margin of the foramen magnum, at its most "
          "posteroinferior point",
    "S": "Sella -- the centre of the sella turcica",
    "N": "Nasion -- the nasofrontal suture, at its most anterosuperior point",
    "A": "A point -- the deepest point of the concavity of the anterior maxilla",
    "ANS": "Anterior nasal spine",
    "PNS": "Posterior nasal spine",
    "B": "B point -- the deepest point of the concavity of the mandibular symphysis",
    "Pog": "Pogonion -- the most anterior point of the mandibular symphysis",
    "Gn": "Gnathion -- on the symphysis, between Pogonion and Menton",
    "Me": "Menton -- the most inferior point of the chin",
    "RCo": "Right condyle -- its superior and central point",
    "LCo": "Left condyle -- its superior and central point",
    "RGo": "Right gonion -- the angle of the mandible",
    "LGo": "Left gonion -- the angle of the mandible",
}


def _describe(label: str) -> str:
    """One line for `label`, or "" when this file cannot source one."""
    if label in _NAMED_POINTS:
        return _NAMED_POINTS[label]
    quadrant, digit = label[:2], label[2:3]
    side = _QUADRANTS.get(quadrant)
    if side is None or not digit.isdigit():
        return ""
    position = int(digit)
    if not 1 <= position <= len(_FROM_MIDLINE):
        return ""
    # Longest first: "MB" and "B" would both match a name ending in "MB".
    point = _POINT_NAMES.get(label[3:])
    if point is None:
        return ""
    boundary, before, onward = point
    return "{} {} -- {}".format(
        side, _FROM_MIDLINE[position - 1],
        onward if position >= boundary else before,
    )


DESCRIPTIONS = {
    label: _describe(label)
    for labels in GROUP_LABELS.values()
    for label in labels
    if _describe(label)
}
