"""The two passes: predict a class per surface, then explain it on the mesh."""

import logging
import os

from . import catalog
from .pipeline import GRADCAM_IMAGE_SIZE, import_torch, scale_attribution

logger = logging.getLogger("DOCShapeAXI")


def _dataset(model, surfaces, mount_point, device):
    """A shapeaxi dataset over the surfaces, built the way upstream builds it.

    shapeaxi takes its file list through a DataFrame, so one is made in memory.
    Upstream wrote a CSV into the OUTPUT folder and read it back -- and opened
    it with mode 'a', so a second run appended to the first run's list.
    """
    import pandas as pd
    from shapeaxi.saxi_dataset import SaxiDataset
    from shapeaxi.saxi_transforms import EvalTransform

    frame = pd.DataFrame({model.hparams.surf_column: [
        os.path.relpath(path, mount_point) for path in surfaces
    ]})
    scale_factor = getattr(model.hparams, "scale_factor", None)
    return SaxiDataset(
        frame,
        transform=EvalTransform(scale_factor),
        CN=True,
        surf_column=model.hparams.surf_column,
        mount_point=mount_point,
        class_column=None,
        scalar_column=None,
    )


def predict(model, analysis, surfaces, mount_point, device) -> list:
    """One prediction per surface, in the order given."""
    import torch
    from torch.utils.data import DataLoader

    dataset = _dataset(model, surfaces, mount_point, device)
    loader = DataLoader(dataset, batch_size=1, pin_memory=False)
    softmax = torch.nn.Softmax(dim=1)

    predictions = []
    with torch.no_grad():
        for vertices, faces, normals in loader:
            vertices = vertices.to(device)
            faces = faces.to(device)
            normals = normals.to(device)

            mesh = model.create_mesh(vertices, faces, normals)
            points = model.sample_points_from_meshes(mesh, model.hparams.sample_levels[0])
            views, _ = model.render(mesh)

            output = model(points, views)
            if not analysis.is_regression:
                # No argmax for a regression checkpoint: its single output IS
                # the value, and taking an argmax of one column returns 0 for
                # every subject.
                output = torch.argmax(softmax(output).detach(), dim=1, keepdim=True)
            predictions.append(float(output.reshape(-1)[0].cpu()))
    return predictions


def explain(model, analysis, surfaces, mount_point, device, output_dir) -> list:
    """One surface per input, carrying a GradCAM array per class.

    Written ONCE per surface, after every class has been added. Upstream wrote
    the file inside the per-class loop, to the same path each time, so a
    four-class run rewrote the same file four times.
    """
    import torch
    from captum.attr import LayerGradCam
    from shapeaxi import post_process, utils
    from shapeaxi.saxi_gradcam import gradcam_process
    from torch.utils.data import DataLoader

    dataset = _dataset(model, surfaces, mount_point, device)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=False)

    blocks = getattr(model.convnet.module, "_blocks")
    cam = LayerGradCam(model, blocks[-1], device_ids=[0])

    written = []
    for index, (vertices, faces, normals) in enumerate(loader):
        vertices = vertices.to(device)
        faces = faces.to(device)
        normals = normals.to(device)

        mesh = model.create_mesh(vertices, faces, normals)
        points = model.sample_points_from_meshes(mesh, model.hparams.sample_levels[0])
        views, per_face = model.render(mesh)

        surface = dataset.getSurf(index)
        source = dataset.getSurfPath(index)

        for class_index in range(analysis.classes):
            attribution = cam.attribute(
                inputs=(points, views), target=class_index, attr_dim_summation=False
            )
            attribution = attribution.sum(dim=1).cpu().detach()
            scaled = scale_attribution(attribution.numpy(), GRADCAM_IMAGE_SIZE)
            projected = gradcam_process(
                _Namespace(device=device), scaled, faces, per_face, vertices, device=device
            )
            surface.GetPointData().AddArray(projected)
            post_process.MedianFilter(surface, projected)

        destination = os.path.join(output_dir, os.path.basename(source))
        utils.WriteSurf(surface, destination)
        written.append(destination)
    return written


class _Namespace:
    """`gradcam_process` reads its arguments off an object; upstream passed the
    whole argparse namespace. Only `device` is read."""

    def __init__(self, device):
        self.device = device
