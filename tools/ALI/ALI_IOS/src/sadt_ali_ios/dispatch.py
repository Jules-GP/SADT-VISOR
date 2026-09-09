"""Everything ALI_IOS does around inference: discovery, the landmark selection
and the run report. `engine.py` only has to know how to place landmarks.

ALI used to be one tool choosing an engine from the data. Splitting it in two
moved that question out of the run and into the request: this tool is the
intraoral engine, and an input holding CBCT volumes is refused by name rather
than half-processed.

No DICOM here, unlike ALI_CBCT: a DICOM series is a volume by definition, so
detecting one is that tool's business. It also needs itk, which this
environment does not carry.
"""

import json
import logging
import os
import shutil
import time

from sadt_ali_common import markups
from sadt_ali_common.discovery import (
    IOS,
    SURFACE_EXTENSIONS,
    VOLUME_EXTENSIONS,
    WORK_DIRNAME,
    classify,
    keyed,
)

from .errors import ToolInputError
from . import catalog as ios_catalog

logger = logging.getLogger(__name__)


REPORT_NAME = "run_report.json"


class Input:
    """What the input turned out to hold, and how to run it.

    `scans` is a list of `(absolute path, key)` pairs. The key is the path
    relative to the input root and is what identifies a scan everywhere
    afterwards -- in the report and in the output tree. Keying by BASE NAME,
    as the original did, meant two patients called `scan.nii.gz` in different
    subfolders silently overwrote each other twice over: once in the working
    dictionary, once in the flat output folder.
    """

    def __init__(self, mode: str, scans: list):
        self.mode = mode
        self.scans = scans



# ---------------------------------------------------------------------------
# Discovery and mode detection
# ---------------------------------------------------------------------------


def detect(input_path: str, work_dir: str) -> Input:
    """List the intraoral surfaces to process, and refuse anything else.

    Before the split this decided WHICH engine ran. It no longer does: this
    tool is the intraoral engine, so the question is only whether the caller
    sent surfaces. Volumes are named rather than ignored -- silently processing
    the meshes of a mixed folder and dropping the scans is the failure that
    looks like success.

    `work_dir` is unused here and kept so both tools' dispatchers have the same
    shape; ALI_CBCT needs it to convert DICOM.

    No archive is unpacked here. The server extracts a `.zip` before `run()`
    is called -- with the bomb cap and the single-root strip this function used
    to apply itself -- so what arrives is always a real file or directory.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    root = input_path

    if os.path.isfile(root):
        lower = root.lower()
        if lower.endswith(SURFACE_EXTENSIONS):
            return Input(IOS, [(root, os.path.basename(root))])
        if lower.endswith(VOLUME_EXTENSIONS):
            raise ToolInputError(
                f"'{os.path.basename(root)}' is a CBCT volume. This tool places "
                f"landmarks on intraoral surfaces; run ALI_CBCT on volumes."
            )
        raise ToolInputError(
            f"'{os.path.basename(root)}' is not an intraoral surface "
            f"({', '.join(SURFACE_EXTENSIONS)})."
        )

    volumes, surfaces = classify(root)

    if volumes and not surfaces:
        raise ToolInputError(
            f"This input holds {len(volumes)} CBCT scan(s) and no intraoral surface. "
            f"Run ALI_CBCT on volumes."
        )
    if volumes:
        raise ToolInputError(
            f"This input mixes {len(volumes)} CBCT scan(s) and {len(surfaces)} intraoral "
            f"surface(s). Send them as two batches, to ALI_CBCT and ALI_IOS respectively."
        )

    if surfaces:
        return Input(IOS, keyed(surfaces, root))

    raise ToolInputError(
        f"No intraoral surface ({', '.join(SURFACE_EXTENSIONS)}) found in the input."
    )


# ---------------------------------------------------------------------------
# Narrowing a run to named landmarks
# ---------------------------------------------------------------------------

def _keep_only(report: dict, landmarks) -> None:
    """Drop every landmark the caller did not name from what was written.

    This is deliberately AFTER the run rather than inside the engine. One
    forward pass covers a (network, jaw, tooth) and emits every channel of that
    network at once, so predicting UR1O predicts UR1MB and UR1DB with it at no
    extra cost: naming landmarks narrows what is WRITTEN, never what is
    computed. What a selection does save is whole passes, and that is decided
    before the run by `catalog.networks_for` -- by the time the engine is
    called there is nothing left for it to narrow.

    Each file is rewritten through the writer that produced it, so a filtered
    file is indistinguishable from one a run of exactly this selection would
    have written, control-point ids included.
    """
    wanted = set(landmarks)
    for record in report.get("scans", {}).values():
        kept_files = []
        for path in record.get("files", []):
            if _filter_markups(path, wanted):
                kept_files.append(path)
            else:
                # Nothing selected survives on this scan -- an upper-arch
                # selection against a mandible, say. An empty markups file
                # opens in Slicer as an empty node, which reads as a run that
                # went wrong; the scan's own record is where "nothing here"
                # belongs.
                os.remove(path)
        record["files"] = kept_files
        record["landmarks_found"] = [
            label for label in record.get("landmarks_found", []) if label in wanted
        ]
        degraded = record.get("landmarks_degraded")
        if degraded:
            record["landmarks_degraded"] = {
                label: note for label, note in degraded.items() if label in wanted
            }


def _filter_markups(path: str, wanted: set) -> list:
    """Rewrite one markups file keeping only `wanted`; return what it kept."""
    with open(path, encoding="utf-8") as handle:
        content = json.load(handle)

    points = [
        point
        for point in content["markups"][0]["controlPoints"]
        if point["label"] in wanted
    ]
    if not points:
        return []

    markups.write(
        {point["label"]: point["position"] for point in points},
        path,
        # A caveat belongs to its point and travels with it: dropping the
        # neighbours it was written beside must not silently make a degraded
        # landmark look like a clean one.
        {
            point["label"]: point["description"]
            for point in points
            if point.get("description")
        },
    )
    return [point["label"] for point in points]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def identify(
    input_path: str,
    model_path: str,
    output_dir: str,
    ios_networks=None,
    landmarks=None,
    prediction_ID: str = "Pred",
    device: str = "cuda",
) -> dict:
    """Place landmarks on whatever this input holds; return the run report.

    Everything is written under `output_dir`: one markups file per scan, in the
    input's own tree, plus `run_report.json`. Intermediates go in
    `<output_dir>/.ali_work/` and are removed before returning.
    """
    started_at = time.monotonic()

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    prediction_ID = (prediction_ID or "Pred").strip() or "Pred"

    try:
        # Walking a cohort and converting DICOM are minutes of work on a large
        # batch, and used to happen in complete silence -- so a run looked hung
        # before it had even started. Counts only, never a file name.
        logger.info("ALI_IOS: inspecting the input")
        detected = detect(input_path, work_dir)
        logger.info("ALI_IOS: %d surface(s)", len(detected.scans))

        # Imported here, not at module level: the engine pulls torch and
        # pytorch3d, and CI imports this package on every PR to publish the
        # schema. That must not cost a CUDA stack.
        from . import engine as ios_engine

        # An explicit landmark list REPLACES the families rather than narrowing
        # them, exactly as it does in ALI_CBCT: a caller naming the points it
        # needs must not also have to leave the right families ticked, and
        # narrowing would silently drop landmarks for one that set both.
        selected, unknown = ios_catalog.resolve_landmarks(landmarks)
        if landmarks and not selected:
            raise ToolInputError(
                f"None of the landmarks named under 'landmarks' exists: "
                f"{', '.join(sorted(set(landmarks)))}. This tool places "
                f"{len(ios_catalog.LANDMARKS)} landmarks, e.g. "
                f"{', '.join(ios_catalog.LANDMARKS[:3])}."
            )

        if selected:
            networks = ios_catalog.networks_for(selected)
        else:
            networks = ios_catalog.network_codes(ios_networks)
            if not networks:
                # The cross-argument rule the schema cannot express.
                raise ToolInputError(
                    f"Select at least one landmark family under 'networks' "
                    f"({', '.join(ios_catalog.NETWORK_NAMES)}), or name the "
                    f"points you want under 'landmarks'."
                )

        report = ios_engine.predict_landmarks(
            meshes=detected.scans,
            model_path=model_path,
            networks=networks,
            prediction_ID=prediction_ID,
            output_dir=output_dir,
            device=device,
        )
        if selected:
            _keep_only(report, selected)
        # What drove the run, and only that: the families are left at their
        # default when landmarks were named, so reporting them would show a
        # selection the caller never made. `networks` above stays as the engine
        # wrote it -- it says which passes actually ran, which is a fact about
        # the run rather than a selection.
        report["landmarks_selected"] = list(selected)
        if unknown:
            # Named rather than dropped in silence: a stale client asking for a
            # landmark this tool no longer places gets a shorter answer than it
            # asked for, and nothing else would say so.
            report["landmarks_unknown"] = list(unknown)
    finally:
        # The intermediates are large -- converted DICOM, and every scan
        # preprocessed at two spacings. Removed whether or not the run
        # succeeded, and never from inside the output tree the caller keeps.
        shutil.rmtree(work_dir, ignore_errors=True)

    report["tool"] = "ALI_IOS"
    # So the report says which weights ran even when nobody read the argument.
    report["model_bundle"] = os.path.basename(str(model_path).rstrip(os.sep))
    report["output_dir"] = output_dir
    report["duration_seconds"] = round(time.monotonic() - started_at, 2)

    # Named `run_report.json` because that is what the Slicer module reads to
    # tell "the model bundle has no such landmark" from "the agent did not
    # converge on this scan" -- two failures that look identical in the scene
    # and need opposite fixes.
    with open(os.path.join(output_dir, REPORT_NAME), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logger.info(
        "ALI_IOS finished: %d/%d scan(s) in %.1fs",
        report["summary"]["processed"],
        report["summary"]["total"],
        report["duration_seconds"],
    )
    return report
