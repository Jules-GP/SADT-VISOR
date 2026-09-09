"""CrownSeg -- per-tooth labelling of intraoral surface scans, via shapeaxi.

The pipeline is in pipeline.py; only `run` is public.
"""

from pathlib import Path
from typing import Literal

from .pipeline import segment_crowns


def run(
    meshes: Path,
    model: Path,
    output_dir: Path,
    array_name: str = "Universal_ID",
    suffix: str = "Seg",
    numbering: Literal["Universal", "FDI"] = "Universal",
    skip_segmented: bool = True,
    device: Literal["cuda", "cpu"] = "cuda",
    num_workers: int = 2,
) -> Path:
    """Label every tooth of an intraoral scan with its dental number.

    Args:
        meshes: One intraoral surface (.vtk/.stl), or a folder of them for a
            batch. Folders are searched recursively and the output mirrors the
            input tree, so two patients whose meshes share a file name stay
            apart.
        model: The crown-segmentation checkpoint (a .pth file), or a folder
            holding it -- the published one is named for the training run that
            produced it, `<date>_val-loss<number>.pth`, and is found inside a
            folder however deeply it is filed. That is what lets a caller hand
            over a whole models directory without knowing where the file sits;
            a folder holding two of them is refused, naming both.
        output_dir: Where the labelled meshes are written, plus
            `run_report.json`. Nothing is written outside it.
        array_name: Name of the point-data array the labels are written to.
        suffix: Added to each output name, e.g. `arch_Seg.vtk`.
        numbering: Universal or FDI tooth numbering. This changes the integers
            written into the array, not the mesh, and whatever consumes the
            result has to agree -- the report records which was used.
        skip_segmented: Pass a mesh that already carries labels through
            unchanged instead of spending minutes re-predicting it. That is
            what makes a mixed batch of raw and pre-segmented meshes one call.
        device: "cuda" or "cpu". CUDA falls back to CPU when no card is
            visible, with a warning.
        num_workers: DataLoader workers shapeaxi uses to load meshes. Forced to
            at least 1: shapeaxi builds its loader with persistent_workers=True,
            which PyTorch rejects at 0.

    Returns:
        The output directory, holding the labelled meshes and the run report.
        The report lists every mesh that now carries labels under
        `segmented_meshes` -- that is what a caller sequencing this before ALI
        reads.
    """
    # shapeaxi, torch and vtk are imported inside the pipeline: CI imports this
    # module on every PR to publish the schema, and the segmentation stack is
    # an optional extra that a CI venv deliberately does not have.
    output_dir = Path(output_dir)
    segment_crowns(
        input_path=str(meshes),
        model_path=str(model),
        output_dir=str(output_dir),
        array_name=array_name,
        suffix=suffix,
        fdi=numbering == "FDI",
        skip_segmented=skip_segmented,
        device=device,
        num_workers=num_workers,
    )
    return output_dir
