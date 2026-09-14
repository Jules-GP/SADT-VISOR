"""Pair an intraoral scan with a CBCT of the same patient, and register it.

NOT longitudinal, unlike its two siblings. AREG_CBCT and AREG_IOS take a
baseline and a follow-up of one modality; this takes ONE timepoint imaged two
ways, and puts the intraoral scan into the CBCT's frame. Upstream's own test
set says so plainly -- `P001_T2_U.vtk` beside `P_0001_T2.nii.gz`, both T2.

That is why the arguments are `ios` and `cbct` rather than `t1` and `t2`, and
why `pairing.pair()` is not what pairs them: there is no timepoint to strip,
only a patient to match across two naming conventions.
"""

import json
import logging
import os

import numpy as np

from sadt_areg_common import pairing
from sadt_areg_common.errors import ToolInputError

from . import geometry

logger = logging.getLogger(__name__)

SURFACE_EXTENSIONS = (".vtk", ".stl")
LANDMARK_EXTENSIONS = (".json", ".mrk.json")


def _patient_key(filename: str) -> str:
    """`P001_T2_U.vtk` and `P_0001_T2.nii.gz` are the same patient.

    The two modalities are named by different conventions -- the intraoral files
    by the scanner, the CBCT by the acquisition -- so the digits are what they
    genuinely share. Everything that is not a digit is dropped and leading zeros
    go with it, which makes `P001` and `P_0001` both `1`.

    Deliberately cruder than `pairing.patient_stem`, and only used here: that
    function matches two files that came from the SAME source and can rely on a
    shared stem. Across modalities there is no shared stem to rely on.
    """
    stem = pairing.split_scan_extension(os.path.basename(filename))[0]
    digits = "".join(character for character in stem if character.isdigit())
    # The trailing timepoint digit is part of the name, not the patient: strip
    # the tokens that name one before reducing to digits.
    tokens = [t for t in pairing.tokens(stem) if t not in ("t0", "t1", "t2")]
    digits = "".join(c for token in tokens for c in token if c.isdigit())
    return digits.lstrip("0") or digits or stem


def discover(ios_dir: str, cbct_dir: str) -> dict:
    """`{patient: {"ios": [paths], "cbct": path}}`, for what is present in both.

    A patient with only one modality is reported rather than silently dropped:
    a batch that registered half of what was sent and said nothing is the
    failure this repository keeps finding.
    """
    ios: dict = {}
    for root, _dirs, files in os.walk(ios_dir):
        for name in sorted(files):
            if name.lower().endswith(SURFACE_EXTENSIONS):
                ios.setdefault(_patient_key(name), []).append(os.path.join(root, name))

    cbct: dict = {}
    for root, _dirs, files in os.walk(cbct_dir):
        for name in sorted(files):
            if pairing.is_scan_file(name):
                cbct[_patient_key(name)] = os.path.join(root, name)

    paired, unpaired = {}, {}
    for key in sorted(set(ios) | set(cbct)):
        if key in ios and key in cbct:
            paired[key] = {"ios": sorted(ios[key]), "cbct": cbct[key]}
        else:
            unpaired[key] = "no CBCT" if key in ios else "no intraoral scan"

    if not paired:
        raise ToolInputError(
            "No patient has both an intraoral scan and a CBCT. Found "
            f"{len(ios)} intraoral key(s) and {len(cbct)} CBCT key(s): {unpaired}."
        )
    if unpaired:
        # The counts, never the keys: a key is the caller's own file name, and
        # this line reaches the server's log. `unpaired` is returned as it is,
        # and the run report names every one of them -- that goes back to
        # whoever sent the data.
        without_cbct = sum(1 for reason in unpaired.values() if reason == "no CBCT")
        logger.warning(
            "Not registered, only one modality present: %d patient(s) with no "
            "CBCT, %d with no intraoral scan",
            without_cbct, len(unpaired) - without_cbct,
        )
    return paired, unpaired


def read_landmarks(path: str) -> dict:
    """`{label: [x, y, z]}` from a Slicer markups file or a plain JSON one.

    Both spellings are in upstream's own test set -- `.mrk.json` for the CBCT
    side, `.json` for the intraoral -- so both are read rather than one being
    declared canonical.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    points = {}
    for markup in payload.get("markups", []):
        for control_point in markup.get("controlPoints", []):
            label = control_point.get("label")
            position = control_point.get("position")
            if label and position:
                points[label] = [float(value) for value in position]
    if points:
        return points

    # The plainer shape: {label: [x, y, z]} at the top level.
    for label, position in payload.items():
        if isinstance(position, (list, tuple)) and len(position) == 3:
            points[label] = [float(value) for value in position]
    return points


def shared_landmarks(moving: dict, fixed: dict) -> tuple:
    """The points both sides name, in one order, plus what was dropped.

    Intersected rather than assumed equal: the two modalities are landmarked by
    different networks, and one missing point on one side used to be an
    IndexError three frames down instead of a line in a report.
    """
    common = sorted(set(moving) & set(fixed))
    dropped = sorted((set(moving) | set(fixed)) - set(common))
    if len(common) < 3:
        raise ToolInputError(
            f"The two modalities share only {len(common)} landmark(s), and an "
            f"alignment needs 3. Intraoral has {sorted(moving)}; CBCT has {sorted(fixed)}."
        )
    return (
        np.array([moving[label] for label in common], dtype=float),
        np.array([fixed[label] for label in common], dtype=float),
        common,
        dropped,
    )


def register_one(mesh_points: np.ndarray, ios_landmarks: dict, cbct_landmarks: dict,
                 cbct_points: np.ndarray = None, max_dist: float = geometry.DEFAULT_MAX_DIST):
    """Align by landmarks, then refine by ICP when there is a surface to refine on.

    The landmark stage alone is what upstream's "Registration" mode does when no
    CBCT surface is available; the ICP is the refinement, and it needs points
    sampled from the CBCT rather than the volume itself.
    """
    moving, fixed, used, dropped = shared_landmarks(ios_landmarks, cbct_landmarks)
    matrix = geometry.align_by_landmarks(mesh_points, moving, fixed)
    report = {"landmarks_used": used, "landmarks_dropped": dropped, "icp": None}

    if cbct_points is not None and len(cbct_points) >= 3:
        moved = geometry.apply(mesh_points, matrix)
        refinement, stats = geometry.icp_point_to_point(moved, cbct_points, max_dist=max_dist)
        matrix = refinement @ matrix
        report["icp"] = stats
    return matrix, report
