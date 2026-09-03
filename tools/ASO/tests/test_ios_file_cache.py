"""The IOS engine reads each file once, and reading it once changes nothing.

`ios.pipeline.FileCache` is a speedup and only a speedup: the meshes and
landmark files a run used to parse three times it now parses once. Two things
have to hold for that to be true, and both are asserted here rather than
argued:

* **the output is byte-identical.** `test_the_cache_changes_no_output_byte`
  runs the same cohort twice -- once with the real cache, once with a cache
  that re-reads every time, which is exactly what the code did before -- and
  compares every file it wrote, byte for byte.
* **nothing writes into a shared mesh.** A cached `vtkPolyData` is handed to
  the registration, to the transform and to the writer; if any of them mutated
  it, the second jaw or the second patient would silently register against a
  moved mesh. `test_the_cache_hands_back_an_unmodified_mesh` reads the file
  again afterwards and compares the points.

And two that make it worth doing: the reference bundle is parsed once for the
whole cohort, not once per jaw per patient, and a patient's meshes are dropped
when that patient is done.
"""

import filecmp
import os

import numpy as np
import pytest
from vtk.util.numpy_support import vtk_to_numpy

from sadt_aso import catalogs, run
from sadt_aso.ios import pipeline as ios_pipeline
from sadt_aso.ios import surfaces

from test_run import _ios_case, _ios_reference, _write_mesh, _UPPER_CENTROIDS


class ReReadingCache(ios_pipeline.FileCache):
    """The reading behaviour of the engine before the cache: every ask is a
    parse. Kept in the tests, not in the tool, because its only use is to show
    that caching moved nothing."""

    def _get(self, read, path, keep):
        return read(path)


def _cohort(root, count, with_markups=False):
    for index in range(count):
        _ios_case(root, key=f"P{index}", with_markups=with_markups)
    return str(root)


def _count_reads(monkeypatch) -> list:
    """Records every path `read_surface` is called with, in order."""
    seen = []
    original = surfaces.read_surface

    def counting(path):
        seen.append(path)
        return original(path)

    monkeypatch.setattr(surfaces, "read_surface", counting)
    return seen


def _tree(root: str) -> list:
    return sorted(
        os.path.relpath(os.path.join(directory, name), root)
        for directory, _, names in os.walk(root)
        for name in names
    )


# ---------------------------------------------------------------------------
# it changes nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "automation", [catalogs.AUTOMATION_FULLY, catalogs.AUTOMATION_SEMI]
)
def test_the_cache_changes_no_output_byte(tmp_path, monkeypatch, automation):
    """The same cohort, oriented with the cache and with a cache that re-reads
    every file exactly as the engine used to. Every written byte must match --
    the meshes, the transforms, the landmarks and the report."""
    semi = automation == catalogs.AUTOMATION_SEMI
    inputs = _cohort(tmp_path / "in", 3, with_markups=semi)
    reference = _ios_reference(tmp_path, with_surfaces=True, with_markups=True)

    outputs = {}
    for label, cache_class in (("cached", ios_pipeline.FileCache), ("plain", ReReadingCache)):
        monkeypatch.setattr(ios_pipeline, "FileCache", cache_class)
        outputs[label] = str(
            run(
                input=inputs,
                reference=reference,
                output_dir=str(tmp_path / label),
                modality=catalogs.MODALITY_IOS,
                automation=automation,
            )
        )

    assert _tree(outputs["cached"]) == _tree(outputs["plain"])
    assert _tree(outputs["cached"])  # the comparison is only worth anything if it ran
    for relative in _tree(outputs["cached"]):
        assert filecmp.cmp(
            os.path.join(outputs["cached"], relative),
            os.path.join(outputs["plain"], relative),
            shallow=False,
        ), relative


def test_the_cache_hands_back_an_unmodified_mesh(tmp_path):
    """A cached mesh is handed to the registration, the transform filter and
    the writer in turn. If any of them wrote into it, the next jaw -- or the
    next patient, for the reference -- would register against moved geometry
    and report success."""
    path = _write_mesh(tmp_path / "M_U_Seg.vtk", _UPPER_CENTROIDS)
    cache = ios_pipeline.FileCache()
    surface = cache.surface(path, keep=True)
    before = vtk_to_numpy(surface.GetPoints().GetData()).copy()

    matrix = np.eye(4)
    matrix[:3, 3] = [10.0, -5.0, 3.0]
    moved = surfaces.transform_surface(surface, matrix)
    surfaces.write_surface(moved, str(tmp_path / "out.vtk"))

    assert np.array_equal(vtk_to_numpy(surface.GetPoints().GetData()), before)
    assert np.array_equal(
        vtk_to_numpy(surfaces.read_surface(path).GetPoints().GetData()), before
    )


# ---------------------------------------------------------------------------
# and it does what it was added for
# ---------------------------------------------------------------------------

def test_the_reference_bundle_is_read_once_for_the_whole_cohort(tmp_path, monkeypatch):
    """It was read once per jaw per patient: a forty-patient batch parsed the
    two gold meshes eighty times."""
    seen = _count_reads(monkeypatch)
    inputs = _cohort(tmp_path / "in", 4)
    reference = _ios_reference(tmp_path)

    run(
        input=inputs,
        reference=reference,
        output_dir=str(tmp_path / "out"),
        modality=catalogs.MODALITY_IOS,
        automation=catalogs.AUTOMATION_FULLY,
    )

    gold = [path for path in seen if os.path.dirname(path) == reference]
    assert len(gold) == 2, gold
    assert sorted(os.path.basename(path) for path in gold) == [
        "Gold_L_Seg.vtk",
        "Gold_U_Seg.vtk",
    ]


def test_a_patients_mesh_is_read_once_per_jaw(tmp_path, monkeypatch):
    """A fully-automated jaw read the patient's mesh twice: once to find the
    transform, once in `_write_jaw` to apply it to the untransformed geometry."""
    seen = _count_reads(monkeypatch)
    inputs = _cohort(tmp_path / "in", 2)
    reference = _ios_reference(tmp_path)

    run(
        input=inputs,
        reference=reference,
        output_dir=str(tmp_path / "out"),
        modality=catalogs.MODALITY_IOS,
        automation=catalogs.AUTOMATION_FULLY,
    )

    patient_reads = [path for path in seen if os.path.dirname(path) == inputs]
    assert len(patient_reads) == len(set(patient_reads)) == 4  # 2 patients x 2 jaws


def test_a_finished_patient_is_not_held_in_memory(tmp_path):
    """Forty parsed intra-oral scans is a gigabyte of a shared server's RAM.
    Only the reference survives a patient."""
    _ios_case(tmp_path / "in", key="P0")
    reference = _ios_reference(tmp_path)
    cache = ios_pipeline.FileCache()
    reference_mesh = cache.surface(
        os.path.join(reference, "Gold_U_Seg.vtk"), keep=True
    )
    patient_mesh = cache.surface(str(tmp_path / "in" / "P0_U_Seg.vtk"))

    cache.release()

    assert cache.surface(os.path.join(reference, "Gold_U_Seg.vtk"), keep=True) \
        is reference_mesh
    assert cache.surface(str(tmp_path / "in" / "P0_U_Seg.vtk")) is not patient_mesh


def test_a_landmark_file_asked_for_twice_is_parsed_once(tmp_path):
    """The semi-automated path reads a jaw's landmarks to register on and reads
    them again to move them into the result."""
    from sadt_aso import markups

    path = str(tmp_path / "P1_U_lm.mrk.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    markups.write_landmarks({"UR6O": np.array([1.0, 2.0, 3.0])}, path)

    cache = ios_pipeline.FileCache()
    first = cache.landmarks(path)
    second = cache.landmarks(path)
    assert list(first) == list(second)
    assert np.array_equal(first["UR6O"], second["UR6O"])
    # A fresh dict each time, over the same arrays: filtering what you were
    # handed must not edit the cache.
    assert first is not second
    assert first["UR6O"] is second["UR6O"]
    first["UR6O"] = np.zeros(3)
    assert np.array_equal(cache.landmarks(path)["UR6O"], [1.0, 2.0, 3.0])
