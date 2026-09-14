"""Builders shared by AutoMatrix's tests.

Nothing here is a stub. AutoMatrix is pure geometry -- SimpleITK reads a
transform, resamples an image and writes it back -- so every volume, every
transform and every markups file these fixtures produce is the real thing,
small enough to be built in `tmp_path` and read back for real.
"""

import json
import os
import sys

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

@pytest.fixture
def volume():
    """Write a small scan with a cube of `value` in the middle."""

    def build(path, value=7, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
              direction=None, dtype=np.int16, size=6):
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.zeros((size, size, size), dtype)
        array[2:4, 2:4, 2:4] = value
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        if direction is not None:
            image.SetDirection(direction)
        sitk.WriteImage(image, str(path))
        return path

    return build


@pytest.fixture
def label_image():
    """A label map holding 1 and 3 and nothing between them.

    The gap is the point: an interpolation that blends produces 2, a label no
    voxel of the input carries.
    """

    def build(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
        array = np.zeros((8, 8, 8), np.int16)
        array[2:5, 2:6, 2:6] = 1
        array[5:8, 2:6, 2:6] = 3
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        return image

    return build


@pytest.fixture
def label_volume(label_image):
    """`label_image`, written to disk."""

    def build(path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(label_image(**kwargs), str(path))
        return path

    return build


@pytest.fixture
def transform_file():
    """An ITK transform, written in ITK's own format."""

    def build(path, translation=(1.0, 0.0, 0.0)):
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(sitk.TranslationTransform(3, translation), str(path))
        return path

    return build


@pytest.fixture
def matrix_file():
    """A bare 4x4 matrix in text -- what greedy writes and calls a `.mat`."""

    def build(path, rows=None, newline_at_end=True):
        rows = rows or [[1, 0, 0, 3.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(" ".join(str(value) for value in row) for row in rows)
        path.write_text(text + ("\n" if newline_at_end else ""))
        return path

    return build


@pytest.fixture
def landmark_file():
    """A minimal markups file: one group, one control point per position."""

    def build(path, points, status="defined"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"markups": [{"controlPoints": [
            {"label": f"L{index}", "position": list(position),
             "positionStatus": status}
            for index, position in enumerate(points)
        ]}]}))
        return path

    return build


@pytest.fixture
def slicer_markups():
    """A markups document shaped like the one ALI and ASO actually write.

    Everything a clinician sees is in here -- the labels, the descriptions and
    the display block -- because none of it is AutoMatrix's to touch.
    """

    def build(path, points=((1.0, 2.0, 3.0),)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/"
                       "Modules/Loadable/Markups/Resources/Schema/"
                       "markups-schema-v1.0.0.json#",
            "markups": [{
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "locked": False,
                "labelFormat": "%N-%d",
                "controlPoints": [
                    {
                        "id": str(index + 1),
                        "label": f"Ba{index}",
                        "description": f"placed at scale {index}",
                        "associatedNodeID": "",
                        "position": list(position),
                        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                        0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": True,
                        "visibility": True,
                        "positionStatus": "defined",
                    }
                    for index, position in enumerate(points)
                ],
                "display": {
                    # TRUE, and it must stay true: false switches the markups
                    # display node off, so Slicer loads the file, builds the
                    # node and draws nothing.
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [0.5, 0.5, 0.5],
                    "glyphScale": 2.0,
                },
            }],
        }, indent=2))
        return path

    return build
