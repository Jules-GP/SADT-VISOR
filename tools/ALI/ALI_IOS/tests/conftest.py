"""Fixtures for ALI_IOS's suite: a mesh, a bundle, and a stubbed network.

The 2D UNet and pytorch3d's rasterizer are the only parts of this tool that
need a card, a checkpoint and a compiled extension. They are stubbed here
rather than skipped, so everything they are wrapped in -- discovery, the naming
rule, the projection of a predicted mask back onto the mesh, the output tree,
the per-mesh failure handling and the run report -- is exercised for real.

The stub is deliberately not a `MagicMock`: it renders a 2x2 image whose pixels
map to KNOWN faces, one of them to `-1` (nothing rendered there). That is what
lets a test assert the two things the projection has to get right -- that a
pixel carrying `-1` is dropped rather than selecting the mesh's last face, and
that the class is taken by argmax over the raw logits.
"""

import os

import pytest
import torch
import vtk


# ---------------------------------------------------------------------------
# Meshes and bundles
# ---------------------------------------------------------------------------

def write_surface(path, labels=(8, 8, 8, 8, 8, 8), array_name="Universal_ID"):
    """A small .vtk polydata carrying one tooth label per point.

    `labels` doubles as the point count: N points on a folded strip, N-2
    triangles. `labels=None` writes a mesh with no label array at all, which is
    what an unsegmented intraoral scan looks like.
    """
    count = 6 if labels is None else len(labels)
    points = vtk.vtkPoints()
    for index in range(count):
        points.InsertNextPoint(float(index), float(index % 2) * 0.5, float(index % 3) * 0.3)

    polys = vtk.vtkCellArray()
    for index in range(count - 2):
        polys.InsertNextCell(3)
        for point_id in (index, index + 1, index + 2):
            polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)

    if labels is not None:
        array = vtk.vtkIntArray()
        array.SetName(array_name)
        for value in labels:
            array.InsertNextValue(int(value))
        surface.GetPointData().AddArray(array)

    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)


def write_bundle(root, names=("Upper_O_model.pth", "Lower_O_model.pth")):
    """A flat IOS bundle: checkpoints that are never opened, only named."""
    root = str(root)
    os.makedirs(root, exist_ok=True)
    for name in names:
        with open(os.path.join(root, name), "wb") as handle:
            handle.write(b"fake checkpoint")
    return root


def tree_of(root):
    """Every file under `root`, as paths relative to it, sorted."""
    return sorted(
        os.path.relpath(os.path.join(directory, name), root)
        for directory, _subdirs, names in os.walk(root)
        for name in names
    )


# ---------------------------------------------------------------------------
# The stubbed network
# ---------------------------------------------------------------------------

# The 2x2 render every stubbed pass produces. Pixel (1, 1) came from no face at
# all, which is what an empty corner of a view looks like.
FACE_OF_PIXEL = ((0, 1), (2, -1))
# Which output channel wins at each pixel, for the crown networks. Channel 0 is
# the background; the landmark types start at 1.
CLASS_OF_PIXEL = ((1, 2), (3, 1))
# The mucogingival network has one landmark class, 1, and one background.
MG_CLASS_OF_PIXEL = ((1, 0), (0, 0))


class _StubMesh:
    """What `pytorch3d.structures.Meshes` is, as far as this engine uses it."""

    def to(self, device):
        return self

    def clone(self):
        return self


class Stubs:
    """The knobs a test turns, and the record of what the engine asked for."""

    def __init__(self):
        self.face_of_pixel = FACE_OF_PIXEL
        self.class_of_pixel = CLASS_OF_PIXEL
        self.mg_class_of_pixel = MG_CLASS_OF_PIXEL
        # Every checkpoint the engine loaded, as (path, device, network code).
        self.loaded = []
        # Every render, as (kind, view count).
        self.rendered = []

    # -- the tensors a render would have produced ---------------------------

    def _pix_to_face(self, views):
        table = torch.tensor(
            [[float(value) for value in row] for row in self.face_of_pixel]
        )
        return table.reshape(1, 1, 2, 2, 1).repeat(views, 1, 1, 1, 1)

    def _images(self, views):
        return torch.zeros(1, views, 4, 2, 2)

    def _logits(self, views, channels, classes):
        logits = torch.zeros(views, channels, 2, 2)
        for row in range(2):
            for column in range(2):
                logits[:, classes[row][column], row, column] = 1.0
        return logits


@pytest.fixture
def stubs(monkeypatch):
    """Replace pytorch3d, the renderer and the UNet; keep everything else real."""
    from sadt_ali_ios import engine as ios_engine
    from sadt_ali_ios import render as render_module

    state = Stubs()

    monkeypatch.setattr(ios_engine, "resolve_device", lambda requested=None: "cpu")
    monkeypatch.setattr(ios_engine, "check_dependencies", lambda: None)
    monkeypatch.setattr(
        render_module,
        "import_pytorch3d",
        lambda: {
            "Meshes": lambda verts, faces, textures: _StubMesh(),
            "TexturesVertex": lambda verts_features: None,
        },
    )
    monkeypatch.setattr(render_module, "build_renderer", lambda device: "renderer")

    def build_network(checkpoint, device, network_code="O"):
        state.loaded.append((checkpoint, device, network_code))

        def unet(batch):
            if network_code == "MG":
                return state._logits(1, 3, state.mg_class_of_pixel)
            return state._logits(batch.shape[0], 4, state.class_of_pixel)

        return unet

    def render_views(renderer, mesh, center, radius, camera_positions, device):
        views = len(camera_positions)
        state.rendered.append(("crown", views))
        return state._images(views), state._pix_to_face(views)

    def render_mg_views(renderer, mesh, aim, directions, radius, device):
        state.rendered.append(("MG", len(directions)))
        return state._images(len(directions)), state._pix_to_face(len(directions))

    monkeypatch.setattr(ios_engine, "_build_network", build_network)
    monkeypatch.setattr(render_module, "render_views", render_views)
    monkeypatch.setattr(render_module, "render_mg_views", render_mg_views)
    return state
