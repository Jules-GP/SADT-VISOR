"""Where a suffix stops being a suffix: the token boundary.

`catalogs.PATIENT_SUFFIXES` are the marks a previous ASO, AMASSS, ALI or AREG
run leaves on a file name, and stripping them is how a subject's several files
are recognised as one subject's. They used to be matched with
`stem.find(suffix)` at any index, which is a substring test wearing a suffix's
clothes:

    P_Seg1_T1.nii.gz -> P
    P_Seg2_T1.nii.gz -> P

-- two subjects with one key, so one of them was silently dropped by the
`setdefault` that builds the pairing, or handed the other's transform.

Every test below is a boundary case of that match: the suffix as a whole token,
as the head of a longer token, as its tail, inside it, repeated, at the start
of a name, as the whole name, with digits after it, in another case, and
combined with a caller's own `also_drop` vocabulary.
"""

import pytest

from sadt_areg_common import catalogs, pairing

JAW = set(catalogs.JAW_TOKENS)
ANATOMY = {
    token for group in catalogs.REGION_TOKENS.values() for token in group
} | set(catalogs.MASK_TOKENS)


# ---------------------------------------------------------------------------
# The suffix as a whole token: still dropped, which is the point of the table.
# ---------------------------------------------------------------------------

def test_a_suffix_that_is_a_whole_token_is_still_dropped():
    """The behaviour the table exists for, and the one a fix must not lose."""
    assert pairing.patient_stem("P1_Seg.nii.gz") == "P1"
    assert pairing.patient_stem("P1_seg.nii.gz") == "P1"
    assert pairing.patient_stem("P1_Scan.nii.gz") == "P1"
    assert pairing.patient_stem("P1_scan.nii.gz") == "P1"
    assert pairing.patient_stem("P1_Or.nii.gz") == "P1"
    assert pairing.patient_stem("P1_OR.nii.gz") == "P1"
    assert pairing.patient_stem("P1_lm.mrk.json") == "P1"


def test_a_multi_token_suffix_is_matched_across_its_separator():
    """`_lm_Pred` is two tokens, and has to match as both or as neither."""
    assert pairing.patient_stem("P1_lm_Pred.mrk.json") == "P1"
    assert pairing.patient_stem("IC_0005_lm_Pred.mrk.json") == "IC_0005"


def test_a_suffix_truncates_rather_than_deletes():
    """Everything the suffix introduces goes with it.

    `A1_seg_CBMASK` is `A1`'s segmentation of the cranial base, not a patient
    called `A1_CBMASK`. Dropping the token alone would invent one.
    """
    assert pairing.patient_stem("A1_seg_CBMASK.nii.gz") == "A1"
    assert pairing.patient_stem("P1_Scan_extra_words.nii.gz") == "P1"


def test_the_longest_suffix_wins_over_a_shorter_one_inside_it():
    """The table is ordered longest first so `_Scanreg` is not cut at `_Scan`.

    With boundary matching the order no longer decides it on its own -- `_Scan`
    inside `_Scanreg` fails the right-hand boundary -- but both roads have to
    lead to the same place.
    """
    assert pairing.patient_stem("P1_Scanreg.nii.gz") == "P1"
    assert pairing.patient_stem("P1_SegOr.nii.gz") == "P1"
    assert pairing.patient_stem("P1_OutReg.nii.gz") == "P1"


# ---------------------------------------------------------------------------
# The suffix as the HEAD of a longer token. This is the defect.
# ---------------------------------------------------------------------------

HEAD_OF_A_LONGER_TOKEN = [
    ("P_Seg1_T1.nii.gz", "P_Seg1"),
    ("P_Seg2_T1.nii.gz", "P_Seg2"),
    ("P_Segovia_T1.nii.gz", "P_Segovia"),
    ("P_Segmentation_T1.nii.gz", "P_Segmentation"),
    ("A_Segmented_T1.nii.gz", "A_Segmented"),
    ("P_Scan1_T1.nii.gz", "P_Scan1"),
    ("P_scanner_T1.nii.gz", "P_scanner"),
    ("P_scanned_T1.nii.gz", "P_scanned"),
    ("P_Orion_T1.nii.gz", "P_Orion"),
    ("P_Orbit_T1.nii.gz", "P_Orbit"),
    ("P_Origin_T1.nii.gz", "P_Origin"),
    ("SMITH_ORTHO_T1.nii.gz", "SMITH_ORTHO"),
    ("SMITH_ORTHOPEDIC_T2.nii.gz", "SMITH_ORTHOPEDIC"),
    ("P_lmk_T1.nii.gz", "P_lmk"),
    ("P_lms_T1.nii.gz", "P_lms"),
    ("P_MERGEDATA_T1.nii.gz", "P_MERGEDATA"),
    ("P_OutRegion_T1.nii.gz", "P_OutRegion"),
    ("P_SegOrbit_T1.nii.gz", "P_SegOrbit"),
]


@pytest.mark.parametrize("filename,key", HEAD_OF_A_LONGER_TOKEN)
def test_a_suffix_at_the_head_of_a_longer_token_is_not_a_suffix(filename, key):
    """`_Seg` opens `_Segovia`, and `Segovia` is somebody's name."""
    assert pairing.patient_stem(filename) == key


def test_two_subjects_whose_names_open_with_a_suffix_keep_two_keys():
    """The failure this whole file is about, stated as the thing that broke.

    Both names used to reduce to `P`, so the pairing's `setdefault` kept
    whichever sorted first and the other subject vanished from a run that
    reported success.
    """
    first = pairing.patient_stem("P_Seg1_T1.nii.gz")
    second = pairing.patient_stem("P_Seg2_T1.nii.gz")
    assert first != second
    # ... and each still pairs with its own second timepoint.
    assert first == pairing.patient_stem("P_Seg1_T2.nii.gz")
    assert second == pairing.patient_stem("P_Seg2_T2.nii.gz")


# ---------------------------------------------------------------------------
# The suffix as the TAIL of, or INSIDE, a longer token.
# ---------------------------------------------------------------------------

def test_a_suffix_at_the_tail_of_a_longer_token_is_not_a_suffix():
    """`SubSeg` ends in the word but is one token, so nothing is dropped."""
    assert pairing.patient_stem("P1_SubSeg.nii.gz") == "P1_SubSeg"
    assert pairing.patient_stem("P1_PreScan.nii.gz") == "P1_PreScan"


def test_a_suffix_inside_a_longer_token_is_not_a_suffix():
    """Neither edge of the match is a separator."""
    assert pairing.patient_stem("P1_PreSegPost.nii.gz") == "P1_PreSegPost"
    assert pairing.patient_stem("P1_rescanned.nii.gz") == "P1_rescanned"


def test_a_hyphen_does_not_open_a_suffix_match():
    """`CBMASK-Seg` is one hyphenated label AMASSS writes, not a `_Seg` mark.

    `-Seg` was never matched, because every entry of the table opens on an
    underscore. Pinned because it is real cohort data: DATA/AREG/testfiles
    holds `C_0001_T1_Or_CBMASK-Seg_Pred.nii.gz`, and it keys to `C_0001`
    through `_Or`, not through the `Seg` further along.
    """
    assert pairing.patient_stem("C_0001_T1_Or_CBMASK-Seg_Pred.nii.gz") == "C_0001"
    # The hyphen survives the match and is then normalised to `_` with every
    # other separator run, which is what `_drop_tokens` has always done.
    assert pairing.patient_stem("P1_CBMASK-Seg.nii.gz") == "P1_CBMASK_Seg"


# ---------------------------------------------------------------------------
# Repeated, at the start, and as the whole name.
# ---------------------------------------------------------------------------

def test_a_repeated_suffix_word_is_cut_at_the_first_real_token():
    """`P1_Seg1_Seg2` is a name; `P1_seg_seg` is a doubly-marked file."""
    assert pairing.patient_stem("P1_Seg1_Seg2_T1.nii.gz") == "P1_Seg1_Seg2"
    assert pairing.patient_stem("P1_seg_seg.nii.gz") == "P1"
    assert pairing.patient_stem("P1_Seg_T1_Seg.nii.gz") == "P1"


def test_a_suffix_word_starting_the_name_is_left_to_the_token_pass():
    """`index > 0` has always guarded this, and it still does.

    A stem that BEGINS with the suffix has no patient before it to keep, so
    truncating would leave the empty string. `_MERGED.nii.gz` is a real
    AMASSS output name and reduces to `MERGED`.
    """
    assert pairing.patient_stem("_MERGED.nii.gz") == "MERGED"
    assert pairing.patient_stem("_Seg_T1.nii.gz") == "Seg"
    assert pairing.patient_stem("_scan.nii.gz") == "scan"


def test_a_suffix_word_with_no_leading_separator_is_just_a_word():
    """`Seg1_T1.vtk` has no `_Seg` in it at all; the file is patient `Seg1`."""
    assert pairing.patient_stem("Seg1_T1.nii.gz") == "Seg1"
    assert pairing.patient_stem("Seg_P1_T1.nii.gz") == "Seg_P1"
    assert pairing.patient_stem("Scan_only.nii.gz") == "Scan_only"


def test_a_suffix_word_that_is_the_entire_name():
    """Nothing to key on, and the answer must still be a string."""
    assert pairing.patient_stem("Seg.nii.gz") == "Seg"
    assert pairing.patient_stem("scan.nii.gz") == "scan"
    assert pairing.patient_stem("_Seg.nii.gz") == "Seg"


# ---------------------------------------------------------------------------
# Digits after a token: the shape of both the defect and the timepoint table.
# ---------------------------------------------------------------------------

def test_a_digit_after_a_suffix_makes_it_part_of_the_identifier():
    assert pairing.patient_stem("P_Seg1.nii.gz") == "P_Seg1"
    assert pairing.patient_stem("P_Seg10_T2.nii.gz") == "P_Seg10"
    assert pairing.patient_stem("P_Or2_T1.nii.gz") == "P_Or2"


def test_a_digit_after_a_timepoint_makes_it_part_of_the_identifier():
    """`T10` is not `T1`, and never was: `also_drop` has always been on tokens.

    Pinned here beside the suffix cases because the two rules now agree, and a
    later simplification that merged them must not reintroduce `T1` matching
    inside `T10`.
    """
    assert pairing.patient_stem("P1_T10.nii.gz") == "P1_T10"
    assert pairing.patient_stem("T1000_T1.nii.gz") == "T1000"
    assert pairing.patient_stem("T10_T1.nii.gz") == "T10"
    assert pairing.patient_stem("PAT1_T1.nii.gz") == "PAT1"


def test_a_jaw_glued_to_a_timepoint_still_splits():
    """`A2_UpperT1.vtk` is upstream's own naming, and the split is narrow.

    It fires only when the head is a known jaw AND the tail a known timepoint,
    so `PAT1` and `A2_UpperX` are untouched. Restated here because the suffix
    fix runs BEFORE this split and must not disturb it.
    """
    assert pairing.patient_stem("A2_UpperT1.vtk", also_drop=JAW) == "A2"
    assert pairing.patient_stem("A2_UpperT2.vtk", also_drop=JAW) == "A2"
    assert pairing.patient_stem("A2_UpperX.vtk", also_drop=JAW) == "A2_UpperX"
    assert pairing.patient_stem("PAT1.vtk", also_drop=JAW) == "PAT1"


# ---------------------------------------------------------------------------
# Case.
# ---------------------------------------------------------------------------

def test_the_two_spellings_the_table_lists_are_both_dropped():
    """`_Seg`/`_seg`, `_Scan`/`_scan` and `_Or`/`_OR` are separate entries."""
    assert pairing.patient_stem("P1_Seg.nii.gz") == "P1"
    assert pairing.patient_stem("P1_seg.nii.gz") == "P1"
    assert pairing.patient_stem("P1_Or.nii.gz") == "P1"
    assert pairing.patient_stem("P1_OR.nii.gz") == "P1"


def test_a_spelling_the_table_does_not_list_is_kept():
    """Suffix matching is case-SENSITIVE, and this fix did not change that.

    `_SEG` and `_sEg` are not in `PATIENT_SUFFIXES`, so they survive as part of
    the identity -- the same answer as before the boundary fix. Widening the
    match to any case is a separate decision with its own cost: `_or` is a
    French and English word, and `MAX_or_MIN` is a plausible label.
    """
    assert pairing.patient_stem("P1_SEG.nii.gz") == "P1_SEG"
    assert pairing.patient_stem("P1_sEg.nii.gz") == "P1_sEg"
    assert pairing.patient_stem("P1_SCAN.nii.gz") == "P1_SCAN"


def test_case_is_preserved_in_what_survives():
    """The key lands in an output path, so a subject keeps the name they got."""
    assert pairing.patient_stem("PatIent_A_T1.nii.gz") == "PatIent_A"
    assert pairing.patient_stem("p_Seg1_t1.nii.gz") == "p_Seg1"


def test_the_case_insensitive_halves_stay_case_insensitive():
    """`also_drop` and the timepoint table match on the LOWERCASED token.

    Unlike the suffix table. Both behaviours predate this fix and both are
    kept: the asymmetry is in `catalogs`, not in the matcher.
    """
    assert pairing.patient_stem("P1_t1.nii.gz") == "P1"
    assert pairing.patient_stem("P1_T1.nii.gz") == "P1"
    assert pairing.patient_stem("P1_MAND_seg.nii.gz", also_drop=ANATOMY) == "P1"
    assert pairing.patient_stem("P1_mand_SEG.nii.gz", also_drop=ANATOMY) == "P1"


# ---------------------------------------------------------------------------
# Combined with `also_drop`.
# ---------------------------------------------------------------------------

def test_also_drop_has_always_matched_whole_tokens():
    """It goes through `_drop_tokens`, which compares one token at a time.

    Pinned rather than assumed: it is half of the requirement, and a later
    rewrite that routed `also_drop` through the suffix path would silently take
    `MAX_01` back to `01` for every caller, not just the jaw one.
    """
    assert pairing.patient_stem("MAXIMUS_01.nii.gz", also_drop=ANATOMY) == "MAXIMUS_01"
    assert pairing.patient_stem("Mandy_3.nii.gz", also_drop=ANATOMY) == "Mandy_3"
    assert pairing.patient_stem("MDX_1.nii.gz", also_drop=ANATOMY) == "MDX_1"
    assert pairing.patient_stem("Transformer_01.nii.gz", also_drop=("transform",)) == \
        "Transformer_01"


def test_a_masks_key_reaches_its_scan_even_when_the_name_opens_with_a_suffix():
    """AREG's `discover_masks` and GreedyReg's copy of it, on a `Seg1` subject.

    The mask and the scan have to land on the same key or the run goes ahead
    unmasked -- which it did, silently.
    """
    scan = pairing.patient_stem("P_Seg1_T1.nii.gz")
    mask = pairing.patient_stem("P_Seg1_T1_MAND_seg.nii.gz", also_drop=ANATOMY)
    assert scan == mask == "P_Seg1"


def test_a_transform_reaches_its_scan_even_when_the_name_opens_with_a_suffix():
    """AutoMatrix's vocabulary, on the same subject."""
    automatrix = ("transform", "matrix", "warp", "reg",
                  "cb", "mand", "max", "md", "mx")
    scan = pairing.patient_stem("P_Seg1_T1.nii.gz")
    assert pairing.patient_stem("P_Seg1_CB_Reg_transform.tfm", also_drop=automatrix) \
        == scan
    assert pairing.patient_stem("P_Seg1_transform.mat", also_drop=("transform",)) == scan


def test_a_landmark_file_reaches_its_scan_even_when_the_name_opens_with_a_suffix():
    """AREG_IOS's `scan_key`: patient AND timepoint, landmark words dropped."""
    landmark = JAW | {"mg", "pred", "lm", "landmarks"}
    scan = pairing.patient_stem("P_Seg1_T1_Lower.vtk",
                                also_drop=landmark, drop_timepoint=False)
    points = pairing.patient_stem("P_Seg1_T1_Lower_MG_Pred.json",
                                  also_drop=landmark, drop_timepoint=False)
    assert scan == points == "P_Seg1_T1"


def test_a_subject_named_after_a_suffix_keeps_its_two_timepoints_apart():
    """`drop_timepoint=False` on the same names, which is the one-scan key."""
    first = pairing.patient_stem("P_Seg1_T1.nii.gz", drop_timepoint=False)
    second = pairing.patient_stem("P_Seg1_T2.nii.gz", drop_timepoint=False)
    assert (first, second) == ("P_Seg1_T1", "P_Seg1_T2")


def test_also_drop_can_remove_a_word_the_suffix_table_will_not_touch():
    """The two passes are independent, and the suffix pass runs first."""
    assert pairing.patient_stem("P_Segovia_T1.nii.gz", also_drop=("segovia",)) == "P"
    assert pairing.patient_stem("P_Orion_T1_Seg.nii.gz", also_drop=("orion",)) == "P"


# ---------------------------------------------------------------------------
# The helper itself, so a failure names the boundary rather than a whole key.
# ---------------------------------------------------------------------------

BOUNDARY_CASES = [
    # stem, suffix, expected index
    ("P1_Seg", "_Seg", 2),
    ("P_Seg1", "_Seg", -1),
    ("P_Seg1_Seg", "_Seg", 6),
    ("P_Segovia", "_Seg", -1),
    ("P_Seg_T1", "_Seg", 1),
    ("Seg_P1", "_Seg", -1),
    ("_Seg", "_Seg", 0),
    ("P1_Seg.extra", "_Seg", 2),
    ("P1_Seg-extra", "_Seg", 2),
    ("P1_Seg extra", "_Seg", 2),
    ("P1_SegExtra", "_Seg", -1),
    ("P1_scanned_scan", "_scan", 10),
]


@pytest.mark.parametrize("stem,suffix,index", BOUNDARY_CASES)
def test_the_boundary_helper_reports_the_first_aligned_occurrence(stem, suffix, index):
    """Every separator the module knows closes a token, not just underscore."""
    assert pairing._token_aligned_index(stem, suffix) == index
