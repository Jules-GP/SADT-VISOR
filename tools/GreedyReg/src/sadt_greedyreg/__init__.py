"""Register two CBCT timepoints with Greedy."""

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal

from sadt_areg_common import pairing

from .pipeline import (
    binarise_mask,
    registration_command,
    resample_command,
    run_greedy,
    write_identity_init,
)

logger = logging.getLogger("GreedyReg")

__all__ = ["run"]


def run(
    t1: Path,
    t2: Path,
    output_dir: Path,
    masks: Path = "",
    initial_transforms: Path = "",
    metric: Literal["NCC", "NMI", "SSD"] = "NCC",
    transform_type: Literal["Rigid", "Affine"] = "Rigid",
    output_suffix: str = "registered",
) -> Path:
    """Register each patient's second CBCT onto their first, with Greedy.

    Args:
        t1: Folder of first-timepoint scans -- the fixed images.
        t2: Folder of second-timepoint scans -- the moving images. Patients are
            paired with T1 by name.
        output_dir: Where the registered volumes and transforms are written.
        masks: Optional folder of masks, one per patient. The registration
            metric is then computed inside the mask only.
        initial_transforms: Optional folder of `.mat` transforms to start from,
            one per patient. Without one, the search starts from identity.
        metric: What the registration optimises. NCC is windowed 4x4x4.
        transform_type: Rigid is 6 degrees of freedom, Affine is 12.
        output_suffix: Appended to each patient's name in the output.

    Returns:
        The output directory, holding `<patient>_<suffix>.nii.gz`,
        `<patient>_transform.mat` and `GreedyReg_report.json`.
    """
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paired by the SHARED rule, not by a regex of this tool's own. Upstream
    # read the patient as `^([A-Za-z]+\d+)` -- letters then digits, with
    # nothing between -- which silently skipped every name this repository's
    # own data uses: `C_0001_T1_Or.nii.gz`, `MAMP_0002_T1.nii.gz`,
    # `IC_0005.nii.gz`. A file that matched nothing was not reported, it was
    # dropped from the dict.
    matched = pairing.pair(str(t1), str(t2), output_suffix)
    report = {
        "tool": "GreedyReg",
        "metric": metric,
        "transform_type": transform_type,
        "output_suffix": output_suffix,
        "unmatched": matched.unmatched_report(),
        "patients": {},
    }

    if not matched:
        raise ValueError(
            "No patient appears in both the T1 and the T2 folder. They are paired "
            "by name, up to the timepoint token -- so 'P1_T1_scan.nii.gz' pairs "
            f"with 'P1_T2.nii.gz'. Found {len(matched.t1_only)} T1-only and "
            f"{len(matched.t2_only)} T2-only patient(s)."
        )

    mask_by_patient = pairing.discover(str(masks), output_suffix) if masks else {}
    init_by_patient = _discover_transforms(str(initial_transforms)) if initial_transforms else {}

    registered = 0
    for patient, files in matched.matched.items():
        fixed, moving = files["t1"], files["t2"]
        entry = {"t1": os.path.basename(fixed), "t2": os.path.basename(moving)}
        scratch = tempfile.mkdtemp(prefix=f"greedyreg_{patient}_")
        try:
            _register_one(
                patient, fixed, moving, output_dir, scratch,
                mask_by_patient.get(patient), init_by_patient.get(patient),
                metric, transform_type, output_suffix, entry,
            )
            registered += 1
        except Exception as exc:
            # Upstream called sys.exit(1) here, so patient 3 failing lost
            # patients 4 to 40 -- and the batch reported nothing about any of
            # them. One patient failing costs one patient.
            logger.exception("GreedyReg failed on %s", patient)
            entry["status"] = "failed"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        report["patients"][patient] = entry

    report["summary"] = {
        "patients": len(report["patients"]),
        "registered": registered,
        "failed": len(report["patients"]) - registered,
    }
    report["duration_seconds"] = round(time.monotonic() - started, 2)

    if not registered:
        raise ValueError(
            "GreedyReg registered none of the pairs it was given. "
            + "; ".join(
                f"{name}: {detail.get('reason', 'unknown')}"
                for name, detail in report["patients"].items()
            )
        )

    (output_dir / "GreedyReg_report.json").write_text(json.dumps(report, indent=2))
    return output_dir


def _discover_transforms(root: str) -> dict:
    """`.mat` files, keyed by the same patient rule the scans are."""
    found = {}
    if not os.path.isdir(root):
        return found
    for directory, _subdirs, names in os.walk(root):
        for name in sorted(names):
            if not name.lower().endswith(".mat") or name.startswith("."):
                continue
            found[pairing.patient_stem(name)] = os.path.join(directory, name)
    return found


def _register_one(patient, fixed, moving, output_dir, scratch, mask, init,
                  metric, transform_type, suffix, entry) -> None:
    """One pair: affine search, then resample the moving image into the fixed."""
    registered_path = output_dir / f"{patient}_{suffix}.nii.gz"
    transform_path = output_dir / f"{patient}_transform.mat"

    if init:
        entry["initial_transform"] = os.path.basename(init)
    else:
        init = os.path.join(scratch, "init.mat")
        write_identity_init(init)

    if mask:
        entry["mask"] = os.path.basename(mask)
        binarised = os.path.join(scratch, "mask.nii.gz")
        binarise_mask(mask, binarised)
        mask = binarised

    run_greedy(registration_command(
        fixed, moving, str(transform_path), init, metric, transform_type, mask or "",
    ))
    run_greedy(resample_command(fixed, moving, str(registered_path), str(transform_path)))

    entry["status"] = "ok"
    entry["transform_maps"] = "the T2 image -> the T1 frame (what greedy -r consumes)"
    entry["outputs"] = [registered_path.name, transform_path.name]
