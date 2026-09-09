"""Segment the impacted canine on a CBCT scan."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

from .pipeline import (
    DEFAULT_SCORE_THRESHOLD,
    build_model,
    discover_scans,
    resolve_device,
    segment_volume,
)

logger = logging.getLogger("CLIC")

__all__ = ["run"]


def run(
    scans: Path,
    model: Path,
    output_dir: Path,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    output_suffix: str = "seg",
    device: Literal["cuda", "cpu"] = "cuda",
) -> Path:
    """Segment the impacted canine on a CBCT scan with a Mask R-CNN.

    Args:
        scans: One CBCT (.nii/.nii.gz), or a folder of them for a batch.
            Folders are searched recursively.
        model: The Mask R-CNN checkpoint (.pth). Named rather than discovered:
            upstream took the alphabetically first `.pth` in a folder, so which
            model vintage ran depended on file names.
        output_dir: Where the segmentations are written. Nothing is written
            outside it.
        score_threshold: Detections scoring below this are not painted. The one
            knob that moves the segmentation, so it is recorded in the report.
        output_suffix: Appended to each scan's name, `<scan>_<suffix>.nii.gz`.
        device: "cuda" or "cpu". CUDA falls back to CPU when no card is visible.

    Returns:
        The output directory, holding one segmentation per scan and
        `CLIC_report.json`.
    """
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    found = discover_scans(str(scans))
    if not found:
        # A batch that could read nothing is a request problem, not a result:
        # the server maps ValueError to 422 with this message, rather than
        # returning a successful run of zero scans.
        raise ValueError(
            f"No .nii or .nii.gz scan found in '{os.path.basename(str(scans))}'. "
            f"CLIC reads NIfTI volumes; NRRD and MetaImage are not supported."
        )

    device = resolve_device(device)
    network, classes = build_model(str(model), device)
    logger.info(
        "CLIC: %d scan(s), %d class(es), device=%s, score_threshold=%.2f",
        len(found), classes, device, score_threshold,
    )

    report = {
        "tool": "CLIC",
        "model": os.path.basename(str(model)),
        "classes": classes,
        "device": device,
        "score_threshold": score_threshold,
        "scans": [],
    }
    written = []

    for path in found:
        entry = {"input": os.path.basename(path)}
        try:
            written.append(
                _segment_one(network, path, output_dir, device,
                             score_threshold, output_suffix, entry)
            )
        except Exception as exc:
            # One unreadable scan must not cost the other 199. Upstream ran one
            # process per scan, so it never had to say this.
            logger.exception("CLIC failed on %s", os.path.basename(path))
            entry["status"] = "failed"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
        report["scans"].append(entry)

    report["summary"] = f"{len(written)}/{len(found)} scan(s) segmented"
    report["duration_seconds"] = round(time.monotonic() - started, 2)

    if not written:
        # The guard counts what was WRITTEN, not what was walked past.
        raise ValueError(
            "CLIC segmented none of the scans it was given. "
            + "; ".join(
                f"{s['input']}: {s.get('reason', 'unknown')}"
                for s in report["scans"] if s.get("status") == "failed"
            )
        )

    (output_dir / "CLIC_report.json").write_text(json.dumps(report, indent=2))
    return output_dir


def _segment_one(network, path, output_dir, device, score_threshold, suffix, entry):
    """One scan, in and out. Returns the path written."""
    import nibabel as nib
    import numpy as np

    image = nib.load(path)
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {volume.ndim}D")

    labels, detections = segment_volume(network, volume, device, score_threshold)

    stem = os.path.basename(path)
    for extension in (".nii.gz", ".nii"):
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
            break

    destination = output_dir / f"{stem}_{suffix}.nii.gz"
    nib.save(nib.Nifti1Image(labels, image.affine, image.header), str(destination))

    entry["status"] = "ok"
    entry["slices"] = int(volume.shape[2])
    entry["detections"] = detections
    entry["labels_present"] = sorted(int(v) for v in np.unique(labels) if v)
    entry["output"] = destination.name
    if not detections:
        # Reported rather than swallowed: a volume the network found nothing in
        # is a legitimate answer AND the signature of a wrong checkpoint, and
        # only the caller can tell them apart.
        entry["note"] = "no detection cleared the score threshold"
    return str(destination)
