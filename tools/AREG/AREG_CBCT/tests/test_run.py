"""AREG_CBCT's unit tests: no GPU, no weights, no network.

Split out of the single `tools/AREG/tests/test_run.py` that AREG had before it
became three tools. That file went on importing `sadt_areg`, which no
virtualenv has provided since -- so it could not be collected, and all 84 of
its tests stopped running without anything failing. Found by running each
tool's suite in its own interpreter.

elastix is fast enough on a 48^3 phantom that the registration itself is tested
rather than mocked.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from sadt_areg_cbct import dispatch, elastix, run, tools
from sadt_areg_cbct import pipeline as cbct_pipeline
from sadt_areg_common import catalogs, pairing
from sadt_areg_common.errors import SupervisorRequired, ToolInputError


class FakeSup:
    """A supervisor, as a tool sees one. Records what it was asked for.

    `outputs` maps a tool name to a callable taking the parameters it was sent
    and returning the directory it "produced", so a test can plant results
    without any of the real tools existing.
    """

    def __init__(self, tmp_path, outputs=None):
        self.out = Path(tmp_path) / "out"
        self.tmp = Path(tmp_path) / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.outputs = outputs or {}
        self.calls = []
        self.messages = []

    def run(self, tool, **params):
        self.calls.append((tool, params))
        maker = self.outputs.get(tool)
        if maker is None:
            raise AssertionError(f"nothing planted for {tool!r} in this test")
        return Path(maker(params))

    def progress(self, fraction, message):
        self.messages.append((fraction, message))

    def log(self, message):
        self.messages.append((None, message))


def _phantom(size=48, seed=0, spacing=0.8, origin=(-140.0, -90.0, 60.0)):
    """A textured volume with an origin far from zero.

    Far from zero on purpose: that is the condition under which elastix's
    centre of rotation matters, and a phantom centred on the origin would let
    the bug this suite pins pass unnoticed.
    """
    rng = np.random.default_rng(seed)
    volume = rng.random((size,) * 3).astype(np.float32) * 120
    zz, yy, xx = np.meshgrid(*[np.arange(size)] * 3, indexing="ij")
    half = size // 2
    volume += 1400 * (
        ((zz - half) ** 2 / 180 + (yy - half + 2) ** 2 / 140 + (xx - half - 2) ** 2 / 160) < 1
    )
    volume += 900 * (
        ((zz - half + 12) ** 2 / 40 + (yy - half - 10) ** 2 / 35 + (xx - half + 12) ** 2 / 30) < 1
    )
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((spacing,) * 3)
    image.SetOrigin(origin)
    return image


def _moved(image, rotation=(0.04, -0.025, 0.03), translation=(1.2, -1.6, 0.9)):
    """`image` displaced by a known rigid transform, and that transform."""
    truth = sitk.Euler3DTransform()
    size = np.array(image.GetSize()) / 2.0
    truth.SetCenter(image.TransformContinuousIndexToPhysicalPoint(size.tolist()))
    truth.SetRotation(*rotation)
    truth.SetTranslation(translation)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(truth.GetInverse())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image), truth


def _write(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(image, path, useCompression=True)
    return path


def _full_mask(image):
    mask = sitk.GetImageFromArray(np.ones(sitk.GetArrayViewFromImage(image).shape, np.uint8))
    mask.CopyInformation(image)
    return mask


def _grid_mesh(rows=12, columns=12, spacing=1.0):
    """A flat triangulated grid, the smallest thing with a real adjacency."""
    points = vtk.vtkPoints()
    for row in range(rows):
        for column in range(columns):
            points.InsertNextPoint(column * spacing, row * spacing, 0.0)

    triangles = vtk.vtkCellArray()
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            for corners in ((a, a + 1, a + columns), (a + 1, a + columns + 1, a + columns)):
                triangle = vtk.vtkTriangle()
                for index, corner in enumerate(corners):
                    triangle.GetPointIds().SetId(index, corner)
                triangles.InsertNextCell(triangle)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(triangles)
    return mesh


class TestMaskDiscovery:
    def test_a_cbct_in_the_name_is_not_a_cranial_base_mask(self, tmp_path):
        """FIX: the region test was `"cb" in basename.lower()`, which makes
        every file whose name contains CBCT a cranial-base mask."""
        image = _phantom(size=16)
        _write(image, str(tmp_path / "P1_CBCT_seg.nii.gz"))
        assert pairing.discover_masks(str(tmp_path), "CB") == {}

    def test_a_mask_keys_to_the_patient_its_scan_keys_to(self, tmp_path):
        image = _phantom(size=16)
        _write(image, str(tmp_path / "masks" / "P1_T1_MAND_seg.nii.gz"))
        found = pairing.discover_masks(str(tmp_path / "masks"), "MAND")
        assert list(found) == ["P1"]

    def test_a_mask_has_to_say_both_what_it_is_and_what_it_covers(self, tmp_path):
        image = _phantom(size=16)
        _write(image, str(tmp_path / "P1_MAND.nii.gz"))       # no seg token
        _write(image, str(tmp_path / "P2_seg.nii.gz"))        # no region token
        _write(image, str(tmp_path / "P3_MAND_seg.nii.gz"))   # both
        assert list(pairing.discover_masks(str(tmp_path), "MAND")) == ["P3"]

    def test_amasss_output_names_are_recognised(self, tmp_path):
        """What the Fully-Automated path actually has to read back."""
        image = _phantom(size=16)
        _write(image, str(tmp_path / "P1_T1_scan_seg_MANDMASK.nii.gz"))
        found = pairing.discover_masks(str(tmp_path), "MAND")
        assert list(found) == ["P1"]


class TestElastix:
    def test_the_centre_of_rotation_is_honoured(self):
        """FIX, and the headline one: `MatrixRetrieval` read elastix's three
        angles and its translation and DROPPED its CenterOfRotationPoint, so
        the SimpleITK transform it built rotated about the physical origin
        instead. The two differ by (I - R)c -- invisible on centred data,
        metres-per-radian off the further the scan sits from the origin.

        Measured here against a known ground truth on a phantom whose origin is
        at (-140, -90, 60) mm: the centre-dropping version lands several
        millimetres out, this one lands within a fifth of a voxel.
        """
        fixed = _phantom(size=48)
        moving, truth = _moved(fixed)

        transform = elastix.register(fixed, moving)

        # The transform elastix returns maps FIXED space to MOVING space, which
        # is the direction sitk's resampler consumes -- and here that is `truth`
        # itself, since `moving` is `fixed` displaced by it.
        probes = [
            fixed.TransformContinuousIndexToPhysicalPoint([float(v) for v in index])
            for index in ([0, 0, 0], [24, 24, 24], [47, 47, 47], [4, 40, 12])
        ]
        errors = [
            np.linalg.norm(
                np.array(transform.TransformPoint(p)) - np.array(truth.TransformPoint(p))
            )
            for p in probes
        ]
        assert max(errors) < 0.2, f"registration is {max(errors):.3f} mm from the truth"

        # And the shipped behaviour, reconstructed, is not: dropping the centre
        # is a real displacement, not a rounding difference.
        without_centre = sitk.Euler3DTransform()
        without_centre.SetRotation(*transform.GetParameters()[:3])
        without_centre.SetTranslation(transform.GetParameters()[3:6])
        dropped = max(
            np.linalg.norm(
                np.array(without_centre.TransformPoint(p)) - np.array(truth.TransformPoint(p))
            )
            for p in probes
        )
        assert dropped > 1.0, "the phantom is too centred for this test to mean anything"

    def test_a_mask_of_a_different_size_is_refused(self):
        """FIX: `fixed_seg.SetOrigin(fixed_image.GetOrigin())` forced the two
        into agreement unconditionally, so a mask that is genuinely a different
        sampling of the patient was applied several millimetres off in
        silence."""
        image = _phantom(size=24)
        mask = _full_mask(_phantom(size=20))
        with pytest.raises(elastix.RegistrationError, match="not the same sampling"):
            elastix.apply_mask(image, mask)

    def test_a_label_the_mask_does_not_hold_is_refused(self):
        """FIX: `if label is not None and label in np.unique(array)` fell
        through to using the WHOLE mask when the label was absent -- asking for
        label 4 of a two-label mask registered on everything and reported
        success."""
        image = _phantom(size=24)
        with pytest.raises(elastix.RegistrationError, match="no label 4"):
            elastix.apply_mask(image, _full_mask(image), label=4)

    def test_a_multi_label_mask_used_whole_says_so(self):
        image = _phantom(size=24)
        array = np.zeros(sitk.GetArrayViewFromImage(image).shape, np.uint8)
        array[4:12] = 1
        array[12:18] = 2
        mask = sitk.GetImageFromArray(array)
        mask.CopyInformation(image)

        _masked, note = elastix.apply_mask(image, mask, label=0)
        assert note and "several labels" in note

    def test_masking_keeps_only_the_masked_region(self):
        image = _phantom(size=24)
        array = np.zeros(sitk.GetArrayViewFromImage(image).shape, np.uint8)
        array[6:14, 6:14, 6:14] = 1
        mask = sitk.GetImageFromArray(array)
        mask.CopyInformation(image)

        masked, _note = elastix.apply_mask(image, mask, label=1)
        kept = sitk.GetArrayViewFromImage(masked)
        assert kept[6:14, 6:14, 6:14].any()
        assert not kept[:6].any() and not kept[14:].any()

    def test_the_masked_image_never_reaches_the_disk(self, tmp_path, monkeypatch):
        """FIX: `MaskedImage` wrote `<temp>/fixed_image_masked.nii.gz` -- one
        FIXED name shared by every patient of a run and by every concurrent
        request being served.

        Run from an empty directory rather than by patching a setting: the
        masked image is held in memory now, so there is no temp directory to
        point anywhere, and "wrote nothing at all" is the stronger claim.
        """
        monkeypatch.chdir(tmp_path)
        fixed = _phantom(size=32)
        moving, _truth = _moved(fixed)
        masked, _note = elastix.apply_mask(fixed, _full_mask(fixed))
        elastix.register(masked, moving)
        assert list(tmp_path.iterdir()) == []


class TestSemiAutomatedCBCT:
    @staticmethod
    def _cohort(tmp_path, subjects=("P1",)):
        for subject in subjects:
            fixed = _phantom(size=48, seed=abs(hash(subject)) % 100)
            moving, _truth = _moved(fixed)
            _write(fixed, str(tmp_path / "T1" / f"{subject}_T1_scan.nii.gz"))
            _write(moving, str(tmp_path / "T2" / f"{subject}_T2_scan.nii.gz"))
            _write(_full_mask(fixed), str(tmp_path / "masks" / f"{subject}_T1_CB_seg.nii.gz"))
        return tmp_path

    def test_a_run_writes_a_registered_scan_a_transform_and_a_report(self, tmp_path):
        self._cohort(tmp_path)
        run = dispatch.register(
            t1_path=str(tmp_path / "T1"),
            t2_path=str(tmp_path / "T2"),
            t1_masks_path=str(tmp_path / "masks"),
            automation=catalogs.AUTOMATION_SEMI,
            regions=["Cranial base"],
            output_dir=str(tmp_path / "out"),
        )
        assert run.succeeded == ["P1"]
        produced = sorted(
            os.path.relpath(os.path.join(directory, name), run.output_dir)
            for directory, _, names in os.walk(run.output_dir)
            for name in names
        )
        assert produced == [
            "AREG_report.json",
            os.path.join("CB", "P1_CB_Reg.nii.gz"),
            os.path.join("CB", "P1_CB_Reg_transform.tfm"),
        ]

        with open(os.path.join(run.output_dir, "AREG_report.json")) as handle:
            report = json.load(handle)
        assert report["summary"] == {"patients": 1, "registered": 1, "failed": 0}
        assert report["patients"]["P1"]["regions"]["CB"]["status"] == "ok"

    def test_the_written_transform_moves_the_t2_onto_the_t1(self, tmp_path):
        """The direction is asserted, not assumed: a transform written the
        other way round still loads and still transforms, so nothing else in
        the archive would show it.

        It is also what the original could NOT give you: it registered against
        a recentred copy of the T2 living in a `<t2>_Center` folder next to the
        caller's own data, and wrote the transform between the T1 and THAT --
        a volume the caller never received.
        """
        self._cohort(tmp_path)
        run = dispatch.register(
            t1_path=str(tmp_path / "T1"),
            t2_path=str(tmp_path / "T2"),
            t1_masks_path=str(tmp_path / "masks"),
            automation=catalogs.AUTOMATION_SEMI,
            regions=["Cranial base"],
            output_dir=str(tmp_path / "out"),
        )
        transform = sitk.ReadTransform(
            os.path.join(run.output_dir, "CB", "P1_CB_Reg_transform.tfm")
        )
        registered = sitk.ReadImage(os.path.join(run.output_dir, "CB", "P1_CB_Reg.nii.gz"))
        moving = sitk.ReadImage(str(tmp_path / "T2" / "P1_T2_scan.nii.gz"))

        # Resampling the ORIGINAL T2 with the written transform reproduces the
        # registered volume the archive holds. It could only do that if the
        # transform lives in the space of the file the caller sent.
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving)
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        reproduced = sitk.Cast(resampler.Execute(moving), sitk.sitkInt16)
        assert np.array_equal(
            sitk.GetArrayViewFromImage(reproduced), sitk.GetArrayViewFromImage(registered)
        )

    def test_a_subject_with_no_mask_is_reported_and_the_batch_goes_on(self, tmp_path):
        self._cohort(tmp_path, subjects=("P1", "P2"))
        os.remove(str(tmp_path / "masks" / "P2_T1_CB_seg.nii.gz"))

        run = dispatch.register(
            t1_path=str(tmp_path / "T1"),
            t2_path=str(tmp_path / "T2"),
            t1_masks_path=str(tmp_path / "masks"),
            automation=catalogs.AUTOMATION_SEMI,
            regions=["Cranial base"],
            output_dir=str(tmp_path / "out"),
        )
        assert run.succeeded == ["P1"]
        failure = run.patients["P2"]["regions"]["CB"]
        assert failure["status"] == "failed"
        assert "no Cranial base mask" in failure["reason"]

    def test_no_pair_at_all_is_a_422_naming_the_pairing_rule(self, tmp_path):
        image = _phantom(size=16)
        _write(image, str(tmp_path / "T1" / "alpha_T1.nii.gz"))
        _write(image, str(tmp_path / "T2" / "beta_T2.nii.gz"))
        with pytest.raises(ToolInputError, match="paired by name"):
            dispatch.register(
                t1_path=str(tmp_path / "T1"),
                t2_path=str(tmp_path / "T2"),
                t1_masks_path=str(tmp_path / "T1"),
                automation=catalogs.AUTOMATION_SEMI,
                regions=["Cranial base"],
                output_dir=str(tmp_path / "out"),
            )


class TestMaskLookup:
    def test_a_mask_under_amasss_own_output_folder_still_finds_its_scan(self, tmp_path):
        """AMASSS writes one `<scan>_<id>_SegOut/` directory per scan, so a mask
        discovered under it keys to `P1_seg_SegOut/P1` while its scan keys to
        `P1`. Without the leaf fallback every Fully-Automated run would report
        'no mask for this subject' for every subject."""
        from sadt_areg_cbct import pipeline as cbct_pipeline

        image = _phantom(size=16)
        _write(
            image,
            str(tmp_path / "P1_T1_scan_seg_SegOut" / "P1_T1_scan_seg_MANDMASK.nii.gz"),
        )
        found = cbct_pipeline.find_masks([str(tmp_path)], "MAND", scan_keys=["P1"])
        assert os.path.basename(found["P1"]) == "P1_T1_scan_seg_MANDMASK.nii.gz"

    def test_an_ambiguous_leaf_is_not_guessed(self, tmp_path):
        """Two subjects genuinely called P1 in different folders must not
        borrow each other's mask."""
        from sadt_areg_cbct import pipeline as cbct_pipeline

        image = _phantom(size=16)
        for site in ("siteA", "siteB"):
            _write(image, str(tmp_path / site / "P1_T1_MAND_seg.nii.gz"))
        found = cbct_pipeline.find_masks([str(tmp_path)], "MAND", scan_keys=["P1"])
        assert "P1" not in found


# Moved from AREG_IOSCBCT's suite: this package is the one that makes
# the call, so this is where a change to it should fail.
def test_the_mask_request_names_amasss_arguments(tmp_path):
    """AREG sends structure CODES. The packaged AMASSS publishes codes as its
    `choices`, so the display-name translation the in-process version needed
    is gone rather than restated -- and this is what says so."""
    planted = tmp_path / "masks"
    planted.mkdir()
    sup = FakeSup(tmp_path, {"AMASSS": lambda params: planted})

    tools.segment_masks(sup, str(tmp_path / "t1"), "/models/AMASSS", ["CBMASK"])

    tool, params = sup.calls[0]
    assert tool == "AMASSS"
    assert params["structures"] == ["CBMASK"]
    # One binary file per structure: `find_masks` looks each region up by
    # name, and a merged multi-label volume makes every region resolve to
    # the same file.
    assert params["merge"] == ["SEPARATE"]
    assert params["generate_surface"] is False
    assert set(params) >= {"scans", "model", "output_dir"}

def test_the_orientation_request_is_the_nested_one(tmp_path):
    """ASO is itself supervised for CBCT, so this is AREG -> ASO -> ALI:
    three tools and three venvs deep. Whatever supplies `sup` supplies the
    callee's too; nothing here arranges that."""
    planted = tmp_path / "oriented"
    planted.mkdir()
    sup = FakeSup(tmp_path, {"ASO": lambda params: planted})

    tools.orient_scans(sup, str(tmp_path / "scans"), "/models/gold", "CBCT")

    tool, params = sup.calls[0]
    assert tool == "ASO"
    assert params["automation"] == "Fully-Automated"
    assert params["modality"] == "CBCT"
    assert params["output_suffix"] == "Or"


# Moved from the single AREG suite when AREG became three tools. These drive
# `main()` with t1/t2, which is this tool's signature, not the orchestrator's;
# they were sitting in AREG_IOSCBCT's file only because the three used to share
# one. The three that crossed `modality` with `automation` are not ported: the
# split replaced that argument with the choice of which tool you call.

class TestArgumentRules:
    def _main(self, tmp_path=None, **overrides):
            """Drive `main()` WITH a supervisor unless a test says otherwise.

            Every server run has one, and these tests are about the argument rules
            rather than about reaching the other tools -- without a supervisor the
            structural refusal fires first and hides the rule under test. The
            absent-supervisor case is `TestTheToolsItDrives`.
            """
            arguments = {
                "automation": catalogs.AUTOMATION_SEMI,
                "t1": "/nonexistent/t1",
                "t2": "/nonexistent/t2",
                "sup": FakeSup(tmp_path or "/tmp/areg-rules"),
            }
            arguments.update(overrides)
            return dispatch.main(**arguments)

    def test_an_empty_region_selection_is_refused(self):
            with pytest.raises(ToolInputError, match="at least one anatomical region"):
                self._main(cbct_regions={name: False for name in catalogs.REGION_CHOICES})

    def test_semi_automated_cbct_without_masks_is_refused(self):
            with pytest.raises(ToolInputError, match="masks you provide"):
                self._main()

    def test_the_oriented_mode_without_a_reference_is_refused(self):
            with pytest.raises(ToolInputError, match="orientation reference"):
                self._main(automation=catalogs.AUTOMATION_ORIENTED)

    def test_a_suffix_that_is_a_path_is_refused(self):
            with pytest.raises(ToolInputError, match="name fragment"):
                self._main(t1_masks="/nonexistent/masks", output_suffix="../escape")

    def test_the_rules_run_before_anything_is_read(self):
            """Every case above passes paths that do not exist. Reaching the file
            system would raise something else entirely, which is the point: an
            unusable request comes back in a second, not after an hour."""
            assert not os.path.exists("/nonexistent/t1")

    def test_a_mode_needing_a_tool_says_so_when_there_is_no_supervisor(self):
            """Nothing about the request is wrong -- there is simply no way to reach
            the other tool. So the message names the mode that DOES work rather than
            an argument to change, because "deploy a tool" is not something the
            person who sent the request can act on."""
            with pytest.raises(SupervisorRequired) as raised:
                tools.require(None, "AMASSS", "Fully-Automated CBCT registration")

            message = str(raised.value)
            assert "Fully-Automated CBCT registration" in message
            assert "AMASSS" in message
            assert "Semi-Automated" in message   # the mode that does work
            assert "t1_masks" in message

    def test_a_supervisor_makes_the_same_mode_acceptable(self):
            assert tools.require(FakeSup("/tmp"), "AMASSS", "anything") is None



def test_fully_automated_without_segmentation_weights_is_refused_up_front():
    """AMASSS receives None otherwise, and fails on

        TypeError: expected str, bytes or os.PathLike object, not NoneType

    fifteen seconds in, inside a child process, reaching the caller as "Tool
    execution failed". The tool is reachable; what is missing is which weights.
    """
    with pytest.raises(ToolInputError, match="segmentation_model"):
        dispatch._check_cbct(
            automation=catalogs.AUTOMATION_FULLY,
            regions=["Cranial base"],
            t1_masks=None,
            reference=None,
            segmentation_model=None,
            sup=FakeSup("/tmp/areg-rules"),
        )


# ---------------------------------------------------------------------------
# Progress -- one region at a time over the whole cohort
# ---------------------------------------------------------------------------

def _events(path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def test_progress_names_the_region_and_counts_the_subjects(tmp_path, monkeypatch):
    """Two loops, nested, and the bar has to cross them once rather than twice.

    Each region is given its own slice of the run: without that the second
    region sends the fraction back to zero, which a watcher reads as a run
    starting over. The region is named because it is anatomy; the subject is
    only ever a number, the key being built from the caller's file names.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    TestSemiAutomatedCBCT._cohort(tmp_path, subjects=("P1", "P2"))

    dispatch.register(
        t1_path=str(tmp_path / "T1"),
        t2_path=str(tmp_path / "T2"),
        t1_masks_path=str(tmp_path / "masks"),
        automation=catalogs.AUTOMATION_SEMI,
        # Only the cranial base has a mask here, so the mandible costs no
        # registration -- the events still have to be there for it.
        regions=["Cranial base", "Mandible"],
        output_dir=str(tmp_path / "out"),
    )

    events = _events(events_file)
    assert [event["message"] for event in events] == [
        "Cranial base: subject 1 of 2", "Cranial base: subject 2 of 2",
        "Mandible: subject 1 of 2", "Mandible: subject 2 of 2",
    ]
    assert [event["fraction"] for event in events] == [0.0, 0.25, 0.5, 0.75]


def test_progress_does_not_travel_through_the_supervisor(tmp_path, monkeypatch):
    """`sup.progress` is not the channel, and that is the point of the design.

    Eleven of the fifteen tools take no supervisor at all -- CONTRIBUTING tells
    authors not to take one they do not need -- so making it the progress
    channel would mean adding a supervisor to nine tools as a conduit. The
    environment variable reaches all fifteen. `FakeSup.messages` records
    everything a tool sends through the supervisor, and this tool sends none of
    its progress that way.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    TestSemiAutomatedCBCT._cohort(tmp_path)
    sup = FakeSup(tmp_path)

    dispatch.register(
        t1_path=str(tmp_path / "T1"),
        t2_path=str(tmp_path / "T2"),
        t1_masks_path=str(tmp_path / "masks"),
        automation=catalogs.AUTOMATION_SEMI,
        regions=["Cranial base"],
        output_dir=str(tmp_path / "out"),
        sup=sup,
    )

    assert sup.messages == []
    assert [event["message"] for event in _events(events_file)] == [
        "Cranial base: subject 1 of 1",
    ]
