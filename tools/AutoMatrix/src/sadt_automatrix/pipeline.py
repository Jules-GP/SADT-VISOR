"""Apply a transform to a scan, a segmentation or a landmark file.

Ported from `Automatrix_CLI/Automatrix_CLI.py`. SimpleITK reads the transform,
resamples the image and writes it back; a landmark file has its points moved by
the INVERSE transform, because a transform that maps image space maps points
the other way.

The interpolation rule is upstream's and is the important one: nearest
neighbour for a segmentation, linear otherwise. Interpolating a label map
linearly invents labels that were never in it.
"""

import json
import logging
import os

logger = logging.getLogger("AutoMatrix")

# Tokens that name what a file IS rather than whose it is, and so must not end
# up in the patient key. The shared `patient_stem` already drops the scan
# suffixes a previous ASO/AMASSS/ALI/AREG run leaves (`_Seg`, `_Or`, `_lm`...);
# these are the transform vocabulary it does not know, plus the region and jaw
# tokens that sit between the patient and the suffix.
#
# Without them `C_0001_CB_Reg_transform.tfm` keys to `C_0001_CB_Reg_transform`
# and matches no scan at all -- which is this tool's whole job.
PATIENT_TOKENS_TO_DROP = (
    "transform", "matrix", "warp", "reg", "cbreg", "mandreg", "maxreg",
    "cb", "mand", "max", "md", "mx",
)


def patient_of(filename: str) -> str:
    """The patient a file belongs to, transform or scan alike.

    Upstream derived this with fifteen chained `.split()` calls and a loop over
    `_T0` to `_T49`, so a patient genuinely named `P_Seg1` was truncated and a
    trailing `.split('.')[0]` broke any name holding a dot.
    """
    from sadt_areg_common import pairing

    return pairing.patient_stem(filename, also_drop=PATIENT_TOKENS_TO_DROP)


# What a transform may be stored as.
TRANSFORM_EXTENSIONS = (".tfm", ".mat", ".h5", ".hdf5", ".txt")

# What may be transformed. `.mrk.json` is a Slicer markups file and is handled
# as points; everything else is read as an image.
IMAGE_EXTENSIONS = (".nii", ".nii.gz", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz")
LANDMARK_EXTENSIONS = (".mrk.json",)


def is_landmark_file(name: str) -> bool:
    return name.lower().endswith(LANDMARK_EXTENSIONS)


def is_image_file(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(IMAGE_EXTENSIONS) and not is_landmark_file(name)


def is_transform_file(name: str) -> bool:
    return name.lower().endswith(TRANSFORM_EXTENSIONS)


def apply_to_landmarks(source: str, transform, destination: str) -> int:
    """Move a Slicer markups file's points. Returns how many moved.

    The INVERSE is applied, as upstream does: a transform produced by a
    registration maps image space, and a point follows the opposite way. A
    transform that cannot be inverted is an error rather than a warning and a
    file copied unchanged -- upstream logged and returned, so the output looked
    like a result.
    """
    with open(source) as handle:
        markups = json.load(handle)

    inverse = transform.GetInverse()

    moved = 0
    for group in markups.get("markups", []):
        for point in group.get("controlPoints", []):
            position = point.get("position")
            if point.get("positionStatus") != "defined":
                continue
            if not isinstance(position, list) or len(position) != 3:
                continue
            point["position"] = list(inverse.TransformPoint(position))
            moved += 1

    with open(destination, "w") as handle:
        json.dump(markups, handle, indent=2)
    return moved


def resample(image, transform, reference=None, is_segmentation: bool = False):
    """The resampled image.

    Nearest neighbour for a segmentation, linear otherwise -- upstream's rule,
    and the one that matters: interpolating a label map linearly produces
    labels that were never in it.

    With no reference, the output keeps the input's grid and only its origin
    moves. That mimics what `ResampleScalarVectorDWIVolume` did without one,
    which is the behaviour the Slicer module was written against.
    """
    import SimpleITK as sitk

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_segmentation else sitk.sitkLinear
    )
    resampler.SetDefaultPixelValue(0)

    if reference is not None:
        resampler.SetReferenceImage(reference)
    else:
        resampler.SetSize(image.GetSize())
        resampler.SetOutputSpacing(image.GetSpacing())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(transform.TransformPoint(image.GetOrigin()))

    return resampler.Execute(image)


def read_transform(path: str):
    """A transform, whichever of the two shapes it was written in.

    ITK's own formats (`.tfm`, `.h5`, and the MATLAB `.mat` ITK writes) are
    read by `sitk.ReadTransform`. Greedy writes something else: a bare 4x4
    matrix, four lines of numbers, and calls it `.mat` -- so does upstream's
    own `writeIdentityInit`. `ReadTransform` refuses it with a MatlabTransformIO
    error, which is why AutoMatrix could not consume what GreedyReg produced.

    The extension does not say which it is, so the plain matrix is the
    fallback rather than a branch on the name.
    """
    import SimpleITK as sitk

    try:
        return sitk.ReadTransform(path)
    except RuntimeError as itk_error:
        matrix = _read_plain_matrix(path)
        if matrix is None:
            raise RuntimeError(
                f"{os.path.basename(path)} is neither an ITK transform nor a "
                f"4x4 matrix in text. ITK said: {itk_error}"
            ) from itk_error
        affine = sitk.AffineTransform(3)
        affine.SetMatrix([value for row in matrix[:3] for value in row[:3]])
        affine.SetTranslation([row[3] for row in matrix[:3]])
        return affine


def _read_plain_matrix(path: str):
    """A 4x4 matrix written as four lines of numbers, or None."""
    try:
        with open(path) as handle:
            rows = [
                [float(value) for value in line.split()]
                for line in handle
                if line.strip()
            ]
    except (ValueError, UnicodeDecodeError):
        return None
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        return None
    return rows
