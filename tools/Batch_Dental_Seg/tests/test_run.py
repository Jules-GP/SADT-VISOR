"""BatchDentalSeg unit tests.

`nnunet_runner.predict_folder` is stubbed, so no GPU, no weights and no network
are needed. Everything around inference runs for real: discovery, the NIfTI
conversion, geometry matching, the output tree, the per-segment split and the
report.
"""

import json
import os

import numpy as np
import pytest
import SimpleITK as sitk

from pathlib import Path

from sadt_batchdentalseg import run
from sadt_batchdentalseg import catalogs, nnunet_runner, pipeline
from sadt_batchdentalseg.errors import ToolInputError


def segmentation_files(report):
    return [path for scan in report["scans"] for path in scan.get("segmentations", [])]


def _write_scan(path: str, size=(8, 8, 8), value: int = 40) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = np.full(size[::-1], value, dtype=np.int16)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 0.5))
    image.SetOrigin((-10.0, -20.0, 30.0))
    sitk.WriteImage(image, path)
    return path


def _model_bundle(root: str, folder: str) -> str:
    """A bundle laid out the way nnUNet needs: the two json files beside a
    fold_0 holding the checkpoint."""
    base = os.path.join(root, folder, "nnUNetTrainer__nnUNetPlans__3d_fullres")
    os.makedirs(os.path.join(base, "fold_0"), exist_ok=True)
    for name in ("dataset.json", "plans.json"):
        with open(os.path.join(base, name), "w", encoding="utf-8") as handle:
            json.dump({}, handle)
    open(os.path.join(base, "fold_0", "checkpoint_final.pth"), "wb").close()
    return base


def _stub_prediction(labels_present):
    """Stand in for nnUNet: write one label volume per input case."""

    def predict_folder(model_folder, input_dir, output_dir, device, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        for name in sorted(os.listdir(input_dir)):
            if not name.endswith("_0000.nii.gz"):
                continue
            case_id = name[: -len("_0000.nii.gz")]
            reference = sitk.ReadImage(os.path.join(input_dir, name))
            array = np.zeros(sitk.GetArrayViewFromImage(reference).shape, dtype=np.uint8)
            # One slice per label, so every requested value is present.
            for index, value in enumerate(labels_present):
                array[index] = value
            mask = sitk.GetImageFromArray(array)
            mask.CopyInformation(reference)
            sitk.WriteImage(mask, os.path.join(output_dir, f"{case_id}.nii.gz"))

    return predict_folder


@pytest.fixture
def stub_nnunet(monkeypatch):
    def _install(labels_present=(1, 2, 3)):
        monkeypatch.setattr(nnunet_runner, "predict_folder", _stub_prediction(labels_present))
        monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested=None: "cpu")

    return _install


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

# The server-side suite also asserted every catalog key appears in
# `scripts/data-manifest.yml`, because a key that drifts from the manifest makes
# an installed model unselectable. That file lives in the server repository and
# a tool package cannot reach it, so the check is gone and the contract is now
# cross-repo: a catalog key here MUST equal the folder the manifest downloads
# that bundle into. Documented in README.md; nothing in this repository can
# enforce it.


def test_label_values_are_unique_within_a_model():
    """Two names sharing one integer would make the split silently write the
    same mask twice under different anatomy."""
    for model in catalogs.MODELS.values():
        values = list(model.labels.values())
        assert len(values) == len(set(values)), model.name


def test_the_universal_model_labels_every_permanent_tooth():
    """1-32 in Universal numbering, which is what downstream tools index by."""
    universal = catalogs.get("UniversalLab")
    assert set(range(1, 33)) <= set(universal.labels.values())
    assert universal.labels["Mandibular canal"] == 55


def test_naso_maxilla_separates_the_maxilla_and_shifts_the_rest():
    """The whole point of that model, and the reason its table is its own: an
    off-by-one here labels the canal as teeth."""
    naso = catalogs.get("NasoMaxillaDentSeg")
    five = catalogs.get("DentalSegmentator")
    assert naso.labels["Maxilla"] == 3
    assert naso.labels["Mandibular canal"] == 6
    assert five.labels["Mandibular canal"] == 5
    assert "Maxilla" not in five.labels


# ---------------------------------------------------------------------------
# Model bundle discovery
# ---------------------------------------------------------------------------

def test_a_bundle_is_found_however_deeply_the_archive_nested_it(tmp_path):
    """DentalSegmentator arrives as a zip with its own Dataset<n>/ tree, the
    other three as flat files: one discovery rule has to cover both."""
    root = str(tmp_path / "models")
    expected = _model_bundle(root, "DentalSegmentator")
    assert nnunet_runner.find_model_folder(os.path.join(root, "DentalSegmentator")) == expected


def test_a_bundle_missing_its_checkpoint_is_not_accepted(tmp_path):
    """A half-downloaded bundle must report 'not installed', not fail inside
    nnUNet's loader."""
    base = _model_bundle(str(tmp_path / "models"), "DentalSegmentator")
    os.remove(os.path.join(base, "fold_0", "checkpoint_final.pth"))
    assert nnunet_runner.find_model_folder(str(tmp_path / "models" / "DentalSegmentator")) is None


def test_an_unusable_bundle_is_an_argument_error_naming_the_setup_command(tmp_path):
    """422, not 500: the request is fine, the deployment's data is not."""
    os.makedirs(str(tmp_path / "models" / "DentalSegmentator"), exist_ok=True)
    with pytest.raises(ToolInputError, match="setup-models.sh"):
        pipeline.resolve_model(str(tmp_path / "models" / "DentalSegmentator"))


def test_a_bundle_whose_name_is_not_a_known_model_is_refused(tmp_path):
    """The label table comes from the bundle's name, so an unrecognised one
    must stop the run: guessing a table would name every structure wrong."""
    _model_bundle(str(tmp_path / "models"), "SomeOtherBundle")
    with pytest.raises(ToolInputError, match="not a BatchDentalSeg model"):
        pipeline.resolve_model(str(tmp_path / "models" / "SomeOtherBundle"))


def test_the_bundle_name_selects_the_label_table(tmp_path):
    """Picking the NasoMaxilla bundle must bring NasoMaxilla's six labels, not
    the five-label table -- they disagree from value 3 onwards."""
    _model_bundle(str(tmp_path / "models"), "NasoMaxillaDentSeg")
    model, _folder = pipeline.resolve_model(
        str(tmp_path / "models" / "NasoMaxillaDentSeg")
    )
    assert model.name == "NasoMaxillaDentSeg"
    assert model.labels["Maxilla"] == 3


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discovery_is_recursive(tmp_path):
    _write_scan(str(tmp_path / "in" / "a" / "scan1.nii.gz"))
    _write_scan(str(tmp_path / "in" / "b" / "deep" / "scan2.nii.gz"))
    found = pipeline.discover_scans(str(tmp_path / "in"), "Seg")
    assert len(found) == 2


def test_a_previous_run_is_not_re_ingested(tmp_path):
    """`scan_Seg.nii.gz` sorts before `scan.nii.gz`, so without this a second
    run would segment the first run's output."""
    _write_scan(str(tmp_path / "in" / "scan.nii.gz"))
    _write_scan(str(tmp_path / "in" / "scan_Seg.nii.gz"))
    found = pipeline.discover_scans(str(tmp_path / "in"), "Seg")
    assert [os.path.basename(path) for path in found] == ["scan.nii.gz"]


def test_an_input_with_no_scan_is_an_argument_error(tmp_path, stub_nnunet):
    stub_nnunet()
    os.makedirs(str(tmp_path / "in"), exist_ok=True)
    with open(str(tmp_path / "in" / "notes.txt"), "w") as handle:
        handle.write("nothing here")
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    with pytest.raises(ToolInputError, match="No scan found"):
        pipeline.segment(
        output_dir=str(tmp_path / "out"),
            input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
        )


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------

def test_a_run_writes_one_segmentation_per_scan_and_a_report(tmp_path, stub_nnunet):
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _write_scan(str(tmp_path / "in" / "p2.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    report = pipeline.segment(
        output_dir=str(tmp_path / "out"),
        input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
    )

    produced = sorted(os.path.basename(path) for path in segmentation_files(report))
    assert produced == ["p1_Seg.nii.gz", "p2_Seg.nii.gz"]
    assert report["summary"] == "2/2 scan(s) segmented"
    assert os.path.isfile(os.path.join(str(tmp_path / "out"), "BatchDentalSeg_report.json"))


def test_the_report_carries_the_label_table(tmp_path, stub_nnunet):
    """The output is a label volume; without the table its integers mean
    nothing to whoever opens it, and the four models disagree on them."""
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "NasoMaxillaDentSeg")

    report = pipeline.segment(
        output_dir=str(tmp_path / "out"),
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "NasoMaxillaDentSeg"),
    )
    assert report["model"] == "NasoMaxillaDentSeg"
    assert report["labels"]["Maxilla"] == 3


def test_the_output_mirrors_the_input_tree(tmp_path, stub_nnunet):
    """Two patients whose scans share a file name must stay apart -- keying on
    the base name would collapse them into one output file."""
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "subjectA" / "scan.nii.gz"))
    _write_scan(str(tmp_path / "in" / "subjectB" / "scan.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    report = pipeline.segment(
        output_dir=str(tmp_path / "out"),
        input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
    )

    relative = sorted(os.path.relpath(path, str(tmp_path / "out")) for path in segmentation_files(report))
    assert relative == [
        os.path.join("subjectA", "scan_Seg.nii.gz"),
        os.path.join("subjectB", "scan_Seg.nii.gz"),
    ]


def test_the_segmentation_lands_on_the_input_scan_geometry(tmp_path, stub_nnunet):
    """A mask whose origin differs from its scan opens offset from the anatomy
    it describes, and nothing in the report would say so."""
    stub_nnunet()
    scan = _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    report = pipeline.segment(
        output_dir=str(tmp_path / "out"),
        input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
    )

    reference = sitk.ReadImage(scan)
    produced = sitk.ReadImage(segmentation_files(report)[0])
    assert produced.GetSize() == reference.GetSize()
    assert np.allclose(produced.GetOrigin(), reference.GetOrigin())
    assert np.allclose(produced.GetSpacing(), reference.GetSpacing())


def test_separate_segments_writes_only_the_labels_actually_present(tmp_path, stub_nnunet):
    """A UniversalLab run would otherwise write 55 files per patient, most of
    them empty -- and an empty mask reads like a structure the model missed."""
    stub_nnunet(labels_present=(1, 3))
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    report = pipeline.segment(
        output_dir=str(tmp_path / "out"),
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        separate_segments=True,
    )

    names = sorted(os.path.basename(path) for path in segmentation_files(report))
    assert names == ["p1_Seg.nii.gz", "p1_Seg_Upper-Skull.nii.gz", "p1_Seg_Upper-Teeth.nii.gz"]


def test_a_separate_segment_holds_only_its_own_label(tmp_path, stub_nnunet):
    stub_nnunet(labels_present=(1, 2))
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    report = pipeline.segment(
        output_dir=str(tmp_path / "out"),
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
        separate_segments=True,
    )

    mandible = next(path for path in segmentation_files(report) if path.endswith("Mandible.nii.gz"))
    # GetArrayFromImage, not GetArrayViewFromImage: a VIEW borrows the image's
    # buffer, and reading it off a temporary the expression drops reads freed
    # memory -- which looks exactly like a corrupt mask.
    values = set(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(mandible))).tolist())
    assert values <= {0, 1}, "a per-segment file is binary"
    assert 1 in values


def test_an_unreadable_scan_does_not_lose_the_others(tmp_path, stub_nnunet, monkeypatch):
    """One corrupt patient in a cohort of forty must not cost the other
    thirty-nine.

    The scan is unreadable at CONVERSION time, which happens before inference:
    an unguarded loop there aborts the whole run before a single scan has been
    segmented, which is what this pins.
    """
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    # A file with a scan extension and no valid volume in it -- exactly what a
    # truncated upload or a mislabelled file looks like.
    with open(str(tmp_path / "in" / "p2.nii.gz"), "wb") as handle:
        handle.write(b"not a volume")
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    report = pipeline.segment(
        output_dir=str(tmp_path / "out"),
        input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
    )

    statuses = {entry["input"]: entry["status"] for entry in report["scans"]}
    assert statuses == {"p1.nii.gz": "ok", "p2.nii.gz": "failed"}
    assert report["summary"] == "1/2 scan(s) segmented"
    assert len(segmentation_files(report)) == 1


def test_a_batch_of_only_unreadable_scans_is_an_argument_error(tmp_path, stub_nnunet):
    """Nothing to infer on: say so rather than hand nnUNet an empty folder and
    report a successful run of zero scans."""
    stub_nnunet()
    os.makedirs(str(tmp_path / "in"), exist_ok=True)
    with open(str(tmp_path / "in" / "p1.nii.gz"), "wb") as handle:
        handle.write(b"not a volume")
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    with pytest.raises(ToolInputError, match="could be read"):
        pipeline.segment(
        output_dir=str(tmp_path / "out"),
            input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
        )


def test_nnunet_case_ids_are_positional_not_patient_names(tmp_path, stub_nnunet, monkeypatch):
    """nnUNet writes its output under the id it was given, so deriving the id
    from the file name would make two `scan.nii.gz` overwrite each other."""
    seen = {}

    def capture(model_folder, input_dir, output_dir, device, **kwargs):
        seen["inputs"] = sorted(os.listdir(input_dir))
        _stub_prediction((1,))(model_folder, input_dir, output_dir, device)

    monkeypatch.setattr(nnunet_runner, "predict_folder", capture)
    monkeypatch.setattr(nnunet_runner, "resolve_device", lambda requested=None: "cpu")

    _write_scan(str(tmp_path / "in" / "a" / "scan.nii.gz"))
    _write_scan(str(tmp_path / "in" / "b" / "scan.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    pipeline.segment(
        output_dir=str(tmp_path / "out"),
        input_path=str(tmp_path / "in"), model_path=str(tmp_path / "models" / "DentalSegmentator")
    )
    assert seen["inputs"] == ["case_0000_0000.nii.gz", "case_0001_0000.nii.gz"]


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def test_run_returns_the_output_directory_it_was_given(tmp_path, stub_nnunet):
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    output = run(
        scans=tmp_path / "in",
        model=tmp_path / "models" / "DentalSegmentator",
        output_dir=tmp_path / "out",
    )

    assert output == tmp_path / "out"
    assert (output / "BatchDentalSeg_report.json").is_file()
    assert (output / "p1_Seg.nii.gz").is_file()


def test_run_writes_nothing_outside_the_output_directory(tmp_path, stub_nnunet):
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")
    before = sorted(p for p in tmp_path.rglob("*") if p.is_file())

    run(scans=tmp_path / "in", model=tmp_path / "models" / "DentalSegmentator",
        output_dir=tmp_path / "out")

    after = sorted(
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_relative_to(tmp_path / "out")
    )
    assert after == before


def test_the_bulky_intermediates_do_not_survive_the_run(tmp_path, stub_nnunet):
    """One converted volume and one predicted volume per scan, inside what
    gets shipped back."""
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    output = run(scans=tmp_path / "in", model=tmp_path / "models" / "DentalSegmentator",
                 output_dir=tmp_path / "out")

    assert not (output / pipeline.WORK_DIRNAME).exists()
    assert sorted(p.name for p in output.iterdir()) == [
        "BatchDentalSeg_report.json",
        "p1_Seg.nii.gz",
    ]


def test_run_accepts_a_single_scan_as_readily_as_a_folder(tmp_path, stub_nnunet):
    stub_nnunet()
    scan = _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    output = run(scans=Path(scan), model=tmp_path / "models" / "DentalSegmentator",
                 output_dir=tmp_path / "out")

    assert (output / "p1_Seg.nii.gz").is_file()


# ---------------------------------------------------------------------------
# the real models
# ---------------------------------------------------------------------------

REAL_MODEL = os.environ.get("SADT_BATCHDENTALSEG_MODEL")
REAL_SCAN = os.environ.get("SADT_BATCHDENTALSEG_SCAN")


@pytest.mark.gpu
@pytest.mark.models
@pytest.mark.skipif(
    not (REAL_MODEL and REAL_SCAN),
    reason="set SADT_BATCHDENTALSEG_MODEL and SADT_BATCHDENTALSEG_SCAN (see tests/data/README.md)",
)
def test_real_model_segments_a_real_scan(tmp_path):
    """A real bundle on a real CBCT, on the GPU.

    Labels are compared against the pre-port implementation separately (see
    README, "Validated against"); what this asserts is that a real checkpoint,
    real scan geometry and the label table survive the repackaging.
    """
    output = run(
        scans=Path(REAL_SCAN),
        model=Path(REAL_MODEL),
        output_dir=tmp_path / "out",
        separate_segments=True,
    )

    with open(output / "BatchDentalSeg_report.json") as handle:
        report = json.load(handle)

    assert report["model"] == os.path.basename(REAL_MODEL)
    assert report["summary"] == "1/1 scan(s) segmented"
    assert report["device"].startswith("cuda")

    labels = next(output.rglob("*_Seg.nii.gz"))
    values = set(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(str(labels)))).tolist())
    assert values - {0}, "the network emitted at least one structure"
    assert values <= {0} | set(report["labels"].values()), "no label outside the table"


# ---------------------------------------------------------------------------
# Progress -- both ends of a run whose middle is opaque
# ---------------------------------------------------------------------------

def test_progress_reports_the_two_ends_and_says_nothing_it_cannot_know(
    tmp_path, stub_nnunet, monkeypatch
):
    """The whole cohort goes to nnUNet in ONE call, so there is no per-scan
    position to report between the reading and the writing.

    The bar therefore stops at 10% for the length of the segmentation. That is
    the truth: interpolating a position from the number of scans would put a
    number on the bar that nothing in the run measured, and the moment one scan
    took twice as long as another it would be wrong in a way nobody could see.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    stub_nnunet()
    _write_scan(str(tmp_path / "in" / "p1.nii.gz"))
    _write_scan(str(tmp_path / "in" / "p2.nii.gz"))
    _model_bundle(str(tmp_path / "models"), "DentalSegmentator")

    pipeline.segment(
        output_dir=str(tmp_path / "out"),
        input_path=str(tmp_path / "in"),
        model_path=str(tmp_path / "models" / "DentalSegmentator"),
    )

    events = [json.loads(line) for line in
              Path(events_file).read_text().splitlines() if line]
    assert [event["message"] for event in events] == [
        "reading scan 1 of 2", "reading scan 2 of 2",
        "segmenting 2 scan(s) in one pass",
        "writing scan 1 of 2", "writing scan 2 of 2",
    ]
    fractions = [event["fraction"] for event in events]
    assert fractions == sorted(fractions)
    # The scan's own name is patient metadata and never travels in a message.
    assert not any("p1" in event["message"] for event in events)
