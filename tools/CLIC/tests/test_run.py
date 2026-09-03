"""CLIC, with the network stubbed: no GPU, no checkpoint, no card."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import sadt_clic
from sadt_clic import pipeline


def _write_scan(path, shape=(8, 8, 3)):
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the network, so everything around it runs for real."""
    def build_model(checkpoint_path, device):
        return object(), 4

    def segment_volume(model, volume, device, score_threshold):
        labels = np.zeros(volume.shape, dtype=np.int16)
        labels[0, 0, 0] = 3
        return labels, 1

    monkeypatch.setattr(sadt_clic, "build_model", build_model)
    monkeypatch.setattr(sadt_clic, "segment_volume", segment_volume)
    monkeypatch.setattr(sadt_clic, "resolve_device", lambda requested: "cpu")


def test_a_folder_is_walked_recursively(tmp_path):
    """Upstream looked one level down and returned the DIRECTORIES that held a
    scan, which the runner then handed to `nib.load`."""
    _write_scan(tmp_path / "a.nii.gz")
    _write_scan(tmp_path / "deep" / "nested" / "b.nii.gz")

    found = pipeline.discover_scans(str(tmp_path))

    assert [os.path.basename(f) for f in found] == ["a.nii.gz", "b.nii.gz"]
    assert all(os.path.isfile(f) for f in found), "a directory reached the loader"


@pytest.mark.parametrize("name,expected", [
    ("scan.nii", True), ("scan.nii.gz", True), ("SCAN.NII.GZ", True),
    # Advertised by upstream's collector and unreadable by nibabel: accepted
    # and then ignored, which is the trap `.stl` fell into in ALI.
    ("scan.nrrd", False), ("scan.mha", False), ("scan.mhd", False),
])
def test_only_the_extensions_nibabel_reads_are_advertised(name, expected):
    assert pipeline.is_scan_file(name) is expected


def test_the_class_count_comes_from_the_checkpoint():
    """Upstream built both heads for 4 classes whatever the file held."""
    import torch

    state = {"roi_heads.box_predictor.cls_score.bias": torch.zeros(7)}
    assert pipeline.class_count(state) == 7

    with pytest.raises(ValueError, match="not a Mask R-CNN"):
        pipeline.class_count({"something.else": torch.zeros(3)})


def test_a_flat_slice_normalises_to_zeros_not_nans():
    flat = np.full((4, 4), 12.0, dtype=np.float32)
    assert not np.isnan(pipeline.normalise(flat)).any()
    assert (pipeline.normalise(flat) == 0).all()


def test_a_batch_reports_every_scan(tmp_path, stubbed):
    _write_scan(tmp_path / "in" / "one.nii.gz")
    _write_scan(tmp_path / "in" / "two.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    report = json.loads((out / "CLIC_report.json").read_text())
    assert report["summary"] == "2/2 scan(s) segmented"
    assert sorted(s["input"] for s in report["scans"]) == ["one.nii.gz", "two.nii.gz"]
    assert all(s["labels_present"] == [3] for s in report["scans"])
    assert (out / "one_seg.nii.gz").exists() and (out / "two_seg.nii.gz").exists()


def test_an_input_with_no_scan_is_refused(tmp_path, stubbed):
    """A batch that could read nothing is a 422, not a successful run of zero
    scans."""
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "notes.txt").write_text("nothing here")

    with pytest.raises(ValueError, match="No .nii or .nii.gz scan"):
        sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                      output_dir=tmp_path / "out", device="cpu")


def test_one_bad_scan_does_not_cost_the_others(tmp_path, stubbed, monkeypatch):
    _write_scan(tmp_path / "in" / "good.nii.gz")
    (tmp_path / "in" / "broken.nii.gz").write_bytes(b"not a nifti")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    report = json.loads((out / "CLIC_report.json").read_text())
    assert report["summary"] == "1/2 scan(s) segmented"
    failed = [s for s in report["scans"] if s["status"] == "failed"]
    assert len(failed) == 1 and "reason" in failed[0]


def test_a_scan_with_no_detection_says_so(tmp_path, stubbed, monkeypatch):
    """An empty segmentation is a legitimate answer AND the signature of a
    wrong checkpoint. Only the caller can tell them apart, so it is reported."""
    monkeypatch.setattr(
        sadt_clic, "segment_volume",
        lambda model, volume, device, threshold: (np.zeros(volume.shape, np.int16), 0),
    )
    _write_scan(tmp_path / "in" / "quiet.nii.gz")

    out = sadt_clic.run(scans=tmp_path / "in", model=tmp_path / "m.pth",
                        output_dir=tmp_path / "out", device="cpu")

    entry = json.loads((out / "CLIC_report.json").read_text())["scans"][0]
    assert entry["status"] == "ok"
    assert entry["detections"] == 0
    assert "no detection" in entry["note"]
