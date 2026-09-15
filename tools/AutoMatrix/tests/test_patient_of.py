"""Which patient a file belongs to, from its name alone.

This is the whole pairing: a transform and the scan it applies to meet only
here. Upstream derived it from fifteen chained `.split()` calls, a loop over
`_T0` to `_T49` and a trailing `.split('.')[0]`; this port asks the shared
`sadt_areg_common.pairing.patient_stem` with AutoMatrix's own vocabulary.
"""

import pytest

from sadt_automatrix.pipeline import PATIENT_TOKENS_TO_DROP, patient_of


# ---------------------------------------------------------------------------
# Every token the tool declares it drops
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token", PATIENT_TOKENS_TO_DROP)
def test_every_declared_token_is_dropped(token):
    """A token names what a file IS, not whose it is. Any one of them left in
    the key makes the transform match no scan, which is this tool's whole job."""
    assert patient_of(f"P1_{token}.tfm") == "P1"


@pytest.mark.parametrize("token", PATIENT_TOKENS_TO_DROP)
def test_every_declared_token_is_dropped_whatever_its_case(token):
    assert patient_of(f"P1_{token.upper()}.tfm") == "P1"


def test_the_readme_example_pairs_a_transform_with_its_scan(tmp_path):
    """`C_0001_CB_Reg_transform.tfm` and `C_0001_T1.nii.gz` are one patient.
    Without the transform vocabulary the first keys to its whole stem and
    matches nothing at all."""
    assert patient_of("C_0001_CB_Reg_transform.tfm") == "C_0001"
    assert patient_of("C_0001_T1.nii.gz") == "C_0001"


# ---------------------------------------------------------------------------
# Where the token sits
# ---------------------------------------------------------------------------

def test_a_token_in_the_middle_is_dropped_and_the_name_closes_up():
    """Not only a suffix: the region and jaw tokens sit between the patient and
    the suffix, and the separators around what is removed collapse to one."""
    assert patient_of("A_reg_B.nii.gz") == "A_B"


def test_a_token_at_the_front_is_dropped_too():
    assert patient_of("CB_P1_T1.nii.gz") == "P1"


def test_several_tokens_at_once():
    assert patient_of("P1_MAND_MAX_CB_reg_matrix_warp_transform.tfm") == "P1"


def test_a_name_that_is_nothing_but_tokens_leaves_nothing():
    """It keys to the empty string rather than to a plausible-looking patient:
    a transform named only for what it is belongs to nobody."""
    assert patient_of("CB_Reg_transform.tfm") == ""


# ---------------------------------------------------------------------------
# What must NOT be dropped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,patient", [
    # Each identifier CONTAINS a token that must not be cut out of it.
    ("Transformer_01.nii.gz", "Transformer_01"),   # transform
    ("MAXIMUS_01.nii.gz", "MAXIMUS_01"),           # max
    ("Registry_7.nii.gz", "Registry_7"),           # reg
    ("MDX_1.nii.gz", "MDX_1"),                     # md
    ("warped_01.nii.gz", "warped_01"),             # warp
    ("Mandy_3.nii.gz", "Mandy_3"),                 # mand
])
def test_a_token_inside_a_real_identifier_is_kept(name, patient):
    """Upstream matched suffixes with `if suffix in os.path.basename(scan)`, so
    `_L` matched anywhere in a name and the dict's iteration order decided which
    transform won. Matching is on whole tokens."""
    assert patient_of(name) == patient


def test_a_name_with_no_token_at_all_is_returned_as_it_stands():
    assert patient_of("patient.nii.gz") == "patient"
    assert patient_of("1234.nii.gz") == "1234"


def test_the_surviving_part_keeps_its_case():
    """The key ends up in output paths, so a patient keeps the name its owner
    gave it."""
    assert patient_of("PatIent_A_transform.tfm") == "PatIent_A"


@pytest.mark.parametrize("name", ["p1_cb_reg.tfm", "P1_CB_REG.tfm", "P1_Cb_ReG.tfm"])
def test_case_does_not_change_which_tokens_are_recognised(name):
    assert patient_of(name).lower() == "p1"


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "P1_T1.nii.gz", "P1_T1.nrrd.gz", "P1_T1.gipl.gz",
    "P1_T1.nii", "P1_T1.nrrd", "P1_T1.gipl",
])
def test_a_compound_extension_is_split_off_whole(name):
    """`.split('.')[0]` would leave `gz` -- or rather cut the name at its first
    dot, which is the next test."""
    assert patient_of(name) == "P1"


def test_a_markups_double_extension_is_split_off_whole():
    assert patient_of("P1_lm.mrk.json") == "P1"
    assert patient_of("P1_reg.mrk.json") == "P1"


def test_a_dot_inside_the_name_does_not_truncate_the_patient():
    """Upstream's trailing `.split('.')[0]` turned `P1.2_scan.nii.gz` into `P1`,
    merging patient 1.2 with patient 1."""
    assert patient_of("P1.2_scan.nii.gz") == "P1_2"
    assert patient_of("P1.2_scan.nii.gz") != "P1"


def test_the_extension_is_split_off_case_insensitively():
    assert patient_of("P1_T1.NII.GZ") == "P1"


# ---------------------------------------------------------------------------
# Suffixes a previous run left, and the timepoint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "P1_T0.nii.gz", "P1_T1.nii.gz", "P1_T2.nii.gz",
])
def test_the_timepoint_is_not_part_of_the_patient(name):
    assert patient_of(name) == "P1"


@pytest.mark.parametrize("name", [
    "P1_Scan.nii.gz", "P1_Seg.nii.gz", "P1_Or.nii.gz", "P1_lm.mrk.json",
    "P1_lm_Pred.mrk.json", "P1_MERGED.nii.gz", "P1_OutReg.nii.gz",
])
def test_a_suffix_a_previous_tool_left_is_not_part_of_the_patient(name):
    """A scan, its AMASSS segmentation, its ASO orientation and its ALI
    landmarks are one patient: this tool is fed exactly what those wrote."""
    assert patient_of(name) == "P1"


def test_a_transform_and_the_three_kinds_of_file_it_applies_to_agree():
    """The one property the whole tool rests on."""
    keys = {
        patient_of("C_0001_T1_scan.nii.gz"),
        patient_of("C_0001_T1_MAND_Seg.nii.gz"),
        patient_of("C_0001_T1_lm.mrk.json"),
        patient_of("C_0001_CB_Reg_transform.tfm"),
    }
    assert keys == {"C_0001"}
