"""Everything FlexReg does before and after the two engines.

Ported from `FlexReg_CLI.py`, whose `main()` was one function branching on a
`type` string over 23 flat parameters read off a Qt form. What is here is the
half that survives without Slicer: read a surface, build a patch on it or
register one surface onto another, write the result.

The coordinate flip is the thing to know. A `.vtk` on disk is LPS and Slicer
works in RAS, so upstream scaled every mesh by (-1, -1, 1) on read and never
undid it: the file it wrote back was RAS. That is invisible while both ends are
Slicer and wrong the moment anything else reads the result, so the flip is
applied on read and again on write, and the mesh that leaves is in the
convention it arrived in. The registration matrix is still reported in the
convention SimpleITK expects, which is what upstream's `flip @ M @ flip` did.
"""

import logging
import os

logger = logging.getLogger(__name__)

SURFACE_EXTENSIONS = (".vtk", ".vtp", ".stl")

# The point-data array each mode registers on. Upstream chose between these two
# with a `type` string; naming them here keeps the choice in one place.
BUTTERFLY_ARRAY = "Butterfly"
MUCOGINGIVAL_ARRAY = "Bottom_MGL"


class ToolInputError(ValueError):
    """The request cannot work, and the message says why to whoever sent it."""


def _flip_lps_ras(surface):
    """Scale by (-1, -1, 1), which converts LPS to RAS and back.

    Its own inverse, so one function serves both directions.
    """
    import vtk

    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)
    flip = vtk.vtkTransformPolyDataFilter()
    flip.SetInputData(surface)
    flip.SetTransform(transform)
    flip.Update()
    return flip.GetOutput()


def read_surface(path):
    """A `.vtk`/`.vtp`/`.stl` surface, in RAS.

    Upstream read `.vtk` only, through `vtkPolyDataReader`, and a `.stl` handed
    to it returned an empty mesh that failed much later with no mention of the
    file. The reader is chosen by extension and an empty result is refused here.
    """
    import vtk

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif suffix == ".stl":
        reader = vtk.vtkSTLReader()
    elif suffix == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        raise ToolInputError(
            "'{}' is not a surface ({}).".format(
                os.path.basename(path), ", ".join(SURFACE_EXTENSIONS))
        )
    reader.SetFileName(path)
    reader.Update()
    surface = reader.GetOutput()
    if surface is None or surface.GetNumberOfPoints() == 0:
        raise ToolInputError(
            "'{}' holds no surface points.".format(os.path.basename(path))
        )
    return _flip_lps_ras(surface)


def write_surface(surface, path):
    """Write back in the convention the input used, binary.

    Binary rather than the writer's ASCII default: it round-trips the float32
    vertices exactly, where ASCII prints about six significant digits, and it
    parses two orders of magnitude faster.
    """
    import vtk

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(path)
    writer.SetFileTypeToBinary()
    writer.SetInputData(_flip_lps_ras(surface))
    writer.Write()
    return path


def surfaces_in(path):
    """Every surface under `path`, or the single file it names.

    A directory is the batch form: the server pays a process start-up per call,
    so a cohort has to arrive as one call rather than one call per patient.
    Sorted, because readdir order varies between filesystems and an unsorted
    walk renames outputs between two runs on the same folder.
    """
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise ToolInputError("Input path does not exist: {}".format(os.path.basename(path)))

    found = []
    for root, _, files in os.walk(path):
        for name in sorted(files):
            if name.lower().endswith(SURFACE_EXTENSIONS):
                found.append(os.path.join(root, name))
    if not found:
        raise ToolInputError(
            "No surface ({}) found in the input.".format(", ".join(SURFACE_EXTENSIONS))
        )
    return sorted(found)


def build_butterfly(surface, teeth, ratios, adjustments, index, shift_lr, shift_ap):
    """Add a `Butterfly<index>` array to `surface`, in place.

    The teeth are Universal numbers and must exist in the mesh's label array;
    `vtkMeanTeeth` raises `ToothNoExist` when one does not, which is the whole
    reason a caller gets a named error instead of an empty patch.
    """
    from .make_butterfly import butterflyPatch
    from .util import ToothNoExist

    try:
        butterflyPatch(
            surf=surface,
            tooth_anterior_right=teeth["anterior_right"],
            tooth_anterior_left=teeth["anterior_left"],
            tooth_posterior_right=teeth["posterior_right"],
            tooth_posterior_left=teeth["posterior_left"],
            ratio_anterior_right=ratios["anterior_right"],
            ratio_anterior_left=ratios["anterior_left"],
            ratio_posterior_left=ratios["posterior_left"],
            ratio_posterior_right=ratios["posterior_right"],
            adjust_anterior_right=adjustments["anterior_right"],
            adjust_anterior_left=adjustments["anterior_left"],
            adjust_posterior_right=adjustments["posterior_right"],
            adjust_posterior_left=adjustments["posterior_left"],
            index=index,
            shift_lr=shift_lr,
            shift_ap=shift_ap,
        )
    except ToothNoExist as missing:
        # Upstream let this reach the CLI's top level, where Slicer showed the
        # traceback. It names a tooth the caller chose, so it is theirs to fix.
        # The engine's own sentence, passed through rather than wrapped: it
        # already reads "This tooth UR6 is not segmented or doesnt exist", and
        # prefixing it produced "This surface has no tooth This tooth UR6...".
        raise ToolInputError(
            "{}. Pick teeth this surface's label array actually carries -- a "
            "lower arch has none of the palate teeth.".format(str(missing).strip())
        )
    return surface


def merge_patches(surface):
    """Collapse `Butterfly1..N` into one `Butterfly` array.

    Upstream did this at the end of every run, so a mesh carrying several drawn
    patches registers on their union.
    """
    import numpy as np
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    point_data = surface.GetPointData()
    merged = None
    index = 1
    while point_data.HasArray("Butterfly{}".format(index)):
        current = vtk_to_numpy(point_data.GetArray("Butterfly{}".format(index)))
        merged = current.copy() if merged is None else np.logical_or(merged, current)
        index += 1

    if merged is None:
        return surface
    array = numpy_to_vtk(merged.astype(np.int32), deep=True)
    array.SetName(BUTTERFLY_ARRAY)
    point_data.AddArray(array)
    surface.Modified()
    return surface


def register(source, target, patch_array):
    """Rigid registration of `source` onto `target`, computed on `patch_array`.

    Returns `(registered surface, 4x4 matrix)`. The matrix is the one applied to
    the source, in the same RAS convention the surfaces are held in.
    """
    import numpy as np
    import vtk

    from .ICP import ICP, vtkICP
    from .vtkSegTeeth import vtkMeshTeeth

    if not source.GetPointData().HasArray(patch_array):
        raise ToolInputError(
            "The moving surface carries no '{}' array, so there is nothing to "
            "register on. Build a patch first.".format(patch_array)
        )

    option = vtkMeshTeeth(list_teeth=[1], property=patch_array)
    result = ICP([vtkICP()], option=option).run(source, target)
    matrix = np.asarray(result["matrix"])

    vtk_matrix = vtk.vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            vtk_matrix.SetElement(row, column, matrix[row, column])

    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)
    applied = vtk.vtkTransformPolyDataFilter()
    applied.SetInputData(source)
    applied.SetTransform(transform)
    applied.Update()
    return applied.GetOutput(), matrix


def write_transform(matrix, path):
    """The registration as a `.tfm` SimpleITK and Slicer both read.

    `flip @ M @ flip` converts the RAS matrix back to LPS, and the result is
    INVERTED because a Slicer transform node maps the space, not the points:
    loading it and applying it to the original must reproduce the registered
    surface. Both steps are upstream's; they are stated here because getting
    either backwards is silent.
    """
    import numpy as np
    import SimpleITK as sitk

    flip = np.diag([-1, -1, 1, 1])
    composed = np.linalg.inv(flip @ matrix @ flip)

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(composed[:3, :3].flatten())
    transform.SetTranslation(composed[:3, 3])
    sitk.WriteTransform(transform, path)
    return path
