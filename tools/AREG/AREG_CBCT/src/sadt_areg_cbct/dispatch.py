"""AREG -- Automated REGistration of two timepoints.

Ported from the Slicer extension's `AREG/` module and its CLI modules
(`AREG_CBCT`, `AREG_IOS`). One tool, two engines, five modes:

|          | Semi-Automated                | Fully-Automated              | Oriented + Fully-Automated |
|----------|-------------------------------|------------------------------|----------------------------|
| **CBCT** | your T1 masks, masked Elastix | AMASSS segments the T1 masks | ASO orients the T1 first   |
| **IOS**  | your segmented meshes         | CrownSeg labels + ASO orients| --                         |

The Slicer envelope is gone entirely: no `<filter-progress>` prints, no
`time.sleep(0.2)` progress theatre, no `sys.exit`, no log file the client
polls, and nothing written into the caller's input tree.

Two entry points, for the same reason AMASSS and ASO have two:

* `register(...)` -> `RegistrationRun`, the real API: the output directory plus
  a structured report. This is what another server-side tool calls.
* `main(...)` -> the output directory's path, the schema adapter `AREG.py` uses.

The Slicer widget built a list of CLI invocations per mode and ran them in
order, passing folders between them. That structure survives, but the steps are
the other packaged tools -- see `tools.py` for how they are called
in-process and through the registry.
"""

import json
import logging
import os
import shutil

from sadt_areg_common.errors import ToolInputError

from sadt_areg_common import catalogs, pairing
from . import dicom, tools

logger = logging.getLogger(__name__)

# This tool IS the modality: it is no longer an argument, so the value the
# report carries and the automation table is keyed by is fixed here.
MODALITY = catalogs.MODALITY_CBCT

REPORT_NAME = "AREG_report.json"

# Intermediates live here, under the output directory the caller owns, and are
# removed before `register` returns. A surviving `.areg_work/` means a run
# crashed.
WORK_DIRNAME = ".areg_work"


class RegistrationRun:
    """Result of `register()`: where the files are, and what actually happened.

    Reported per patient AND per region, because a CBCT run registering on the
    cranial base and the mandible is two registrations of every patient and one
    of them can fail on its own. The original caught each per-patient exception
    into a log line and finished by printing a count; the archive said nothing.
    """

    def __init__(self, output_dir: str, report: dict):
        self.output_dir = output_dir
        self.report = report

    @property
    def patients(self) -> dict:
        return self.report["patients"]

    @property
    def succeeded(self) -> list:
        return [key for key, entry in self.patients.items() if entry.get("status") == "ok"]



def _check_cbct(automation: str, regions: list, t1_masks, reference, sup=None) -> None:
    if not regions:
        raise ToolInputError(
            "Select at least one anatomical region to register on in 'cbct_regions' "
            f"({', '.join(catalogs.REGION_CHOICES)}). Each one is a separate "
            "registration with its own output folder."
        )

    if automation == catalogs.AUTOMATION_SEMI:
        if not t1_masks:
            raise ToolInputError(
                "Semi-Automated CBCT registers inside masks you provide: send the T1 "
                "segmentations in 't1_masks', or use Fully-Automated mode to have "
                "them produced server-side."
            )
        return

    # Both automated modes need the segmentation; the oriented one also needs
    # the orientation. Checked before the input is extracted -- with the tool
    # absent, the answer is the same whatever the rest of the request says.
    tools.require(sup, "AMASSS", f"{automation} CBCT registration")
    if automation == catalogs.AUTOMATION_ORIENTED:
        tools.require(sup, "ASO", "Oriented + Fully-Automated CBCT registration")
        if not reference:
            raise ToolInputError(
                "Oriented + Fully-Automated CBCT orients the T1 scans before "
                "registering onto them, which needs an orientation reference: name "
                "one in 'cbct_reference' (see GET /tools/AREG/data)."
            )


def _run_cbct(
    t1_root, t2_root, t1_masks_path, automation, regions, segmentation_model,
    segmentation_label, orientation_reference, dicom_input, output_dir, work_dir,
    suffix, report, sup=None, landmark_model=None,
) -> None:
    # Imported here rather than at module level: the CBCT engine pulls in
    # SimpleITK and itk-elastix, and AREG must load on a server without them so
    # its schema is still published and its IOS mode still runs.
    from . import elastix
    from . import pipeline as cbct_pipeline

    elastix.check_dependencies()

    if dicom_input:
        t1_root = dicom.convert_tree(t1_root, os.path.join(work_dir, "dicom_t1"))
        t2_root = dicom.convert_tree(t2_root, os.path.join(work_dir, "dicom_t2"))

    codes = [catalogs.region_code(name) for name in regions]
    report["regions"] = list(regions)
    report["segmentation_label"] = segmentation_label or None

    # Step 1 -- orient the T1 scans, when the mode asks for it. The T2 is NOT
    # oriented: it is about to be resampled into the T1's frame anyway, and
    # orienting it first would be one more interpolation of the same data.
    if automation == catalogs.AUTOMATION_ORIENTED:
        oriented = tools.orient_scans(
            sup,
            t1_root, orientation_reference, catalogs.MODALITY_CBCT,
            landmark_model=landmark_model or "",
        )
        report["oriented_t1"] = True
        t1_root = oriented

    # Step 2 -- the masks the registration is confined to.
    mask_roots = []
    if t1_masks_path:
        mask_roots.append(_as_directory(t1_masks_path, os.path.join(work_dir, "masks_input")))
    if automation == catalogs.AUTOMATION_SEMI:
        # Where the original looked when no mask folder was given.
        mask_roots.append(t1_root)
    else:
        structures = [catalogs.REGION_MASK_STRUCTURES[code] for code in codes]
        mask_roots.append(
            tools.segment_masks(sup, t1_root, segmentation_model, structures)
        )
        report["segmented_t1"] = sorted(structures)

    # Step 3 -- pair the timepoints, then register once per region.
    matched = pairing.pair(t1_root, t2_root, suffix)
    report["unmatched"] = matched.unmatched_report()
    if not matched:
        raise ToolInputError(
            "No subject appears in both the T1 and the T2 folder. They are paired by "
            "name, up to the timepoint token and a trailing "
            f"{', '.join(catalogs.PATIENT_SUFFIXES[:4])}... -- so 'P1_T1_scan.nii.gz' "
            f"in one folder pairs with 'P1_T2.nii.gz' in the other. Found "
            f"{len(matched.t1_only)} T1-only and {len(matched.t2_only)} T2-only subject(s)."
        )

    for code in codes:
        masks = cbct_pipeline.find_masks(mask_roots, code, scan_keys=matched.matched)
        for key, entry in sorted(matched.matched.items()):
            record = report["patients"].setdefault(key, {"status": "ok", "regions": {}})
            mask_path = masks.get(key)
            if not mask_path:
                record["regions"][code] = {
                    "status": "failed",
                    "reason": _no_mask_reason(automation, code),
                }
                continue
            try:
                record["regions"][code] = cbct_pipeline.register_patient(
                    t1_path=entry["t1"],
                    t2_path=entry["t2"],
                    mask_path=mask_path,
                    region=code,
                    output_dir=output_dir,
                    relative_key=key,
                    suffix=suffix,
                    segmentation_label=segmentation_label or None,
                )
            except elastix.RegistrationError as exc:
                record["regions"][code] = {"status": "failed", "reason": str(exc)}
            except RuntimeError as exc:
                record["regions"][code] = {"status": "failed", "reason": f"registration failed: {exc}"}

    _roll_up_regions(report["patients"])


def _no_mask_reason(automation: str, code: str) -> str:
    region = catalogs.region_name(code)
    if automation == catalogs.AUTOMATION_SEMI:
        return (
            f"no {region} mask for this subject. A mask is matched to its scan by name "
            f"and has to say both that it is a segmentation (mask/seg/pred) and which "
            f"structure it covers ({'/'.join(catalogs.REGION_TOKENS[code][:2])}) -- "
            f"e.g. 'P1_T1_{code}_seg.nii.gz' next to 'P1_T1_scan.nii.gz'"
        )
    return (
        f"the segmentation step produced no {region} mask for this subject -- see the "
        f"AMASSS report if one was included in this archive"
    )


def _roll_up_regions(patients: dict) -> None:
    """A patient is 'ok' when at least one of its regions registered."""
    for entry in patients.values():
        statuses = [region.get("status") for region in entry["regions"].values()]
        entry["status"] = "ok" if "ok" in statuses else "failed"


def _selected(value, choices: dict) -> list:
    """The enabled options of a multichoice argument, in declaration order.

    Accepts the `Selection` validate() produces, a plain dict, or a sequence --
    so `register()` stays directly callable with `["Mandible"]`.
    """
    if value is None:
        return [name for name, on in choices.items() if on]
    if isinstance(value, dict):
        return [name for name in choices if value.get(name)]
    wanted = set(value)
    return [name for name in choices if name in wanted]


def _merge_into(source: str, destination: str) -> None:
    """Copy every file of `source` under `destination`, keeping its tree.

    The two timepoints' landmarks end up in ONE folder on purpose: they are
    matched to their scan by a key that carries the timepoint, so they cannot
    collide, and one folder is one index for the painter to search.
    """
    for directory, _, file_names in os.walk(source):
        relative = os.path.relpath(directory, source)
        target = os.path.join(destination, "" if relative == "." else relative)
        os.makedirs(target, exist_ok=True)
        for file_name in file_names:
            shutil.copy2(os.path.join(directory, file_name), os.path.join(target, file_name))


def _as_directory(path: str, destination: str) -> str:
    """A directory holding the input, whatever shape it arrived in.

    A single uploaded file is linked into a directory of its own rather than
    used from where it landed: main.py streams every upload of a request into
    ONE work directory, so treating a file's parent as an input root would make
    the T2 folder part of the T1 one.
    """
    path = str(path)
    if os.path.isdir(path):
        return path

    os.makedirs(destination, exist_ok=True)
    linked = os.path.join(destination, os.path.basename(path))
    try:
        os.link(path, linked)
    except OSError:
        shutil.copy2(path, linked)
    return destination


def _summarize(report: dict) -> None:
    statuses = [entry.get("status") for entry in report["patients"].values()]
    report["summary"] = {
        "patients": len(statuses),
        "registered": statuses.count("ok"),
        "failed": statuses.count("failed"),
    }
    logger.info(
        "AREG %s %s: %d/%d registered",
        report["modality"],
        report["automation"],
        report["summary"]["registered"],
        report["summary"]["patients"],
    )


def register(
    t1_path: str,
    t2_path: str,
    automation: str,
    regions=None,
    t1_masks_path: str = None,
    segmentation_model: str = None,
    segmentation_label: int = 0,
    orientation_reference: str = None,
    landmark_model: str = None,
    dicom_input: bool = False,
    output_suffix: str = "Reg",
    output_dir: str = None,
    sup=None,
) -> RegistrationRun:
    """Register every T2 under `t2_path` onto its T1 under `t1_path`.

    Each path is a directory or a `.zip`. `regions` are the display names
    declared in `catalogs.REGION_CHOICES` (CBCT only).
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    t1_root = _as_directory(t1_path, os.path.join(work_dir, "t1_input"))
    t2_root = _as_directory(t2_path, os.path.join(work_dir, "t2_input"))

    report = {
        "modality": MODALITY,
        "automation": automation,
        "output_suffix": output_suffix,
        "patients": {},
    }

    _run_cbct(
        t1_root=t1_root,
        t2_root=t2_root,
        t1_masks_path=t1_masks_path,
        automation=automation,
        regions=list(regions or ()),
        segmentation_model=segmentation_model,
        segmentation_label=int(segmentation_label or 0),
        orientation_reference=orientation_reference,
        dicom_input=dicom_input,
        output_dir=output_dir,
        work_dir=work_dir,
        suffix=output_suffix,
        report=report,
        sup=sup,
        landmark_model=landmark_model,
    )

    # Extracted inputs, converted DICOM, the oriented copies and whatever the
    # tools it drove wrote. Removed whether or not the run succeeded, so what is
    # left under output_dir is results and nothing else.
    shutil.rmtree(work_dir, ignore_errors=True)

    _summarize(report)
    with open(os.path.join(output_dir, REPORT_NAME), "w") as handle:
        json.dump(report, handle, indent=2)
    return RegistrationRun(output_dir, report)


def main(
    automation,
    t1,
    t2,
    t1_masks=None,
    cbct_regions=None,
    segmentation_label=0,
    segmentation_model=None,
    cbct_reference=None,
    landmark_model=None,
    dicom_input=False,
    output_suffix="Reg",
    output_dir=None,
    sup=None,
) -> str:
    """Translate the schema's arguments into `register()` and return its output
    directory, which main.py zips and streams.

    Every cross-argument rule is checked HERE, before any file is read: a
    request that cannot work must come back in a second, not after an hour of
    registration. `require` in tools.py is part of that: a mode that needs
    another tool fails at the door when there is no supervisor to reach it.
    """
    automation = str(automation)
    suffix = (output_suffix or "Reg").strip() or "Reg"
    if os.sep in suffix or (os.altsep and os.altsep in suffix):
        raise ToolInputError("'output_suffix' is a name fragment, not a path.")

    allowed = catalogs.AUTOMATION_BY_MODALITY.get(MODALITY, ())
    if automation not in allowed:
        raise ToolInputError(
            f"'{automation}' is not a mode {MODALITY} has. {MODALITY} offers: "
            f"{', '.join(allowed)}."
        )

    regions = _selected(cbct_regions, catalogs.REGION_CHOICES)
    reference = cbct_reference
    _check_cbct(automation, regions, t1_masks, reference, sup)

    run = register(
        t1_path=str(t1),
        t2_path=str(t2),
        automation=automation,
        regions=regions,
        t1_masks_path=str(t1_masks) if t1_masks else None,
        segmentation_model=str(segmentation_model) if segmentation_model else None,
        segmentation_label=int(segmentation_label or 0),
        orientation_reference=str(reference) if reference else None,
        landmark_model=landmark_model,
        dicom_input=bool(dicom_input),
        output_suffix=suffix,
        output_dir=output_dir,
        sup=sup,
    )

    return run.output_dir
