"""Apply a transform to scans, segmentations and landmark files."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

from .pipeline import (
    apply_to_landmarks,
    patient_of,
    is_image_file,
    is_landmark_file,
    is_transform_file,
    read_transform,
    resample,
)

logger = logging.getLogger("AutoMatrix")

__all__ = ["run"]


def run(
    files: Path,
    transforms: Path,
    output_dir: Path,
    reference: Path = "",
    content: Literal["Scan", "Segmentation"] = "Scan",
    name_output_after_transform: bool = False,
    output_suffix: str = "Reg",
) -> Path:
    """Apply each patient's transform to their scans, segmentations or landmarks.

    Args:
        files: A scan, a segmentation, a landmark file, or a folder of them.
            Folders are searched recursively.
        transforms: The transforms to apply (.tfm/.mat/.h5/.txt), matched to
            files by patient name. A patient with several transforms has each
            of them applied.
        output_dir: Where the results are written. The input tree is mirrored.
        reference: Optional volume defining the output grid. Without it each
            image keeps its own grid and only its origin moves.
        content: "Segmentation" resamples with nearest neighbour, so no label
            is invented; "Scan" resamples linearly.
        name_output_after_transform: Add the transform's name to each output,
            which is what tells two transforms of one patient apart.
        output_suffix: Appended to each output name.

    Returns:
        The output directory, holding the transformed files and
        `AutoMatrix_report.json`.
    """
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = _discover(str(files))
    if not subjects:
        raise ValueError(
            f"No scan, segmentation or landmark file found in "
            f"'{os.path.basename(str(files))}'."
        )

    by_patient = _discover_transforms(str(transforms))
    if not by_patient:
        raise ValueError(
            f"No transform found in '{os.path.basename(str(transforms))}'. "
            f"Expected one of {', '.join(('.tfm', '.mat', '.h5', '.txt'))}."
        )

    reference_image = None
    if reference:
        import SimpleITK as sitk

        reference_image = sitk.ReadImage(str(reference))

    report = {
        "tool": "AutoMatrix",
        "content": content,
        "reference": os.path.basename(str(reference)) if reference else None,
        "output_suffix": output_suffix,
        "patients": {},
        "without_a_transform": sorted(set(subjects) - set(by_patient)),
        "transforms_without_a_file": sorted(set(by_patient) - set(subjects)),
    }

    written = []
    for patient in sorted(subjects):
        matrices = by_patient.get(patient)
        if not matrices:
            # Named rather than skipped in silence: upstream dropped a file
            # whose patient had no transform without a word, so a run could
            # transform 3 of 40 and look complete.
            continue
        entry = {"transforms": [os.path.basename(m) for m in matrices], "outputs": []}
        for path in subjects[patient]:
            for matrix in matrices:
                try:
                    written.append(_apply_one(
                        path, matrix, files, output_dir, reference_image,
                        content == "Segmentation", name_output_after_transform,
                        output_suffix, entry,
                    ))
                except Exception as exc:
                    logger.exception("AutoMatrix failed on %s", os.path.basename(path))
                    entry.setdefault("failed", []).append(
                        f"{os.path.basename(path)}: {type(exc).__name__}: {exc}"
                    )
        report["patients"][patient] = entry

    report["summary"] = {
        "patients": len(report["patients"]),
        "files_written": len(written),
    }
    report["duration_seconds"] = round(time.monotonic() - started, 2)

    if not written:
        raise ValueError(
            "AutoMatrix transformed nothing. "
            f"{len(report['without_a_transform'])} file(s) had no transform, "
            f"{len(report['transforms_without_a_file'])} transform(s) had no file."
        )

    (output_dir / "AutoMatrix_report.json").write_text(json.dumps(report, indent=2))
    return output_dir


def _discover(root: str) -> dict:
    """{patient: [paths]} for everything transformable under `root`."""
    found: dict = {}
    if os.path.isfile(root):
        name = os.path.basename(root)
        if is_image_file(name) or is_landmark_file(name):
            found.setdefault(patient_of(name), []).append(root)
        return found

    for directory, _subdirs, names in os.walk(root):
        for name in sorted(names):
            if name.startswith(".") or not (is_image_file(name) or is_landmark_file(name)):
                continue
            found.setdefault(patient_of(name), []).append(
                os.path.join(directory, name)
            )
    return found


def _discover_transforms(root: str) -> dict:
    """{patient: [transform paths]}, keyed by the same rule the files are."""
    found: dict = {}
    if os.path.isfile(root):
        if is_transform_file(os.path.basename(root)):
            found.setdefault(patient_of(os.path.basename(root)), []).append(root)
        return found

    for directory, _subdirs, names in os.walk(root):
        for name in sorted(names):
            if name.startswith(".") or not is_transform_file(name):
                continue
            found.setdefault(patient_of(name), []).append(
                os.path.join(directory, name)
            )
    return found


def _apply_one(path, matrix, input_root, output_dir, reference, is_segmentation,
               name_after_transform, suffix, entry) -> str:
    """One file through one transform. Returns the path written."""
    import SimpleITK as sitk

    transform = read_transform(matrix)
    name = os.path.basename(path)

    tail = suffix
    if name_after_transform:
        tail = f"{suffix}_{Path(matrix).stem}"

    relative = os.path.relpath(path, str(input_root)) if os.path.isdir(str(input_root)) else name
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)

    if is_landmark_file(name):
        stem = name[: -len(".mrk.json")]
        destination = destination.parent / f"{stem}_{tail}.mrk.json"
        moved = apply_to_landmarks(path, transform, str(destination))
        entry["outputs"].append({"file": destination.name, "points_moved": moved})
        return str(destination)

    for extension in (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl"):
        if name.lower().endswith(extension):
            stem, tail_extension = name[: -len(extension)], extension
            break
    else:
        stem, tail_extension = os.path.splitext(name)

    destination = destination.parent / f"{stem}_{tail}{tail_extension}"
    image = sitk.ReadImage(path)
    sitk.WriteImage(resample(image, transform, reference, is_segmentation), str(destination))
    entry["outputs"].append({"file": destination.name})
    return str(destination)
