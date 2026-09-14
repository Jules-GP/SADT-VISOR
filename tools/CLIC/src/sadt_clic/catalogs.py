"""What CLIC's classes MEAN, and how a viewer should paint them.

The network emits integers, and here the integer IS the finding: buccal,
bicortical and palatal is the CLASSIFICATION half of "Classification and
Localization of Impacted Canines" -- where the impacted canine sits relative to
the alveolar cortical plates, which is what decides the surgical approach. A
result without this table cannot be read at all.

Taken from upstream's `CLIC.py::_legend`, the only place in that repository
where the three classes are named: it painted a legend over the slice views and
nothing else recorded the mapping -- not the runner, not the file it wrote, not
the module's own help text.

Upstream's legend also coloured segments by CREATION ORDER rather than by label
value (`cols.get(i + 1)`), so a scan where only the palatal class was detected
got a single segment, indexed 1, painted green and read as buccal. Everything
here is keyed by the value the network emits, which is the only thing that
cannot drift.
"""

# {class name: the integer the network emits}, background excluded. Same shape
# as BatchDentalSeg's tables, so a client has one kind of label table to read.
LABELS = {
    "Buccal": 1,
    "Bicortical": 2,
    "Palatal": 3,
}

# RGB in 0-1, which is both the range Slicer's colour API takes and the form
# upstream wrote these exact values in.
COLORS = {
    "Buccal": (0.0, 1.0, 0.0),
    "Bicortical": (1.0, 1.0, 0.0),
    "Palatal": (0.6, 0.4, 0.2),
}

# The class count these names describe, background included. It is what the
# published checkpoint holds, and what `pipeline.class_count` reads back from
# any other one.
CLASSES = len(LABELS) + 1


def labels_for(classes: int):
    """The table, or None when the checkpoint is not the one it describes.

    A checkpoint trained on a different number of classes emits integers these
    names do not cover, and a wrong table does not fail -- it renames the
    finding, which is the one failure nobody downstream can see. Publishing
    nothing is the only honest answer.
    """
    return dict(LABELS) if classes == CLASSES else None


def colors_for(classes: int):
    """`{class name: [r, g, b]}` beside `labels_for`, or None on the same rule."""
    if labels_for(classes) is None:
        return None
    return {name: list(color) for name, color in COLORS.items()}


def names_for(values, classes: int) -> list:
    """`[3]` -> `["Palatal"]`. Empty when the table does not apply."""
    table = labels_for(classes)
    if table is None:
        return []
    by_value = {value: name for name, value in table.items()}
    return [by_value[value] for value in values if value in by_value]
