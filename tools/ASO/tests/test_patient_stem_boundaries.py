"""Where a suffix stops being a suffix: the token boundary.

`cbct.pipeline.PATIENT_SUFFIXES` are the marks a previous ASO, AMASSS or ALI
run leaves on a file name, and stripping them is how a scan and its landmark
file are recognised as one patient's. They used to be matched with
`stem.find(suffix)` at any index, which is a substring test wearing a suffix's
clothes: every entry opens on an underscore, so the match was anchored on the
left and anchored nowhere on the right.

    SMITH_ORTHO_Scan.nii.gz    -> SMITH
    SMITH_ORLANDO_Scan.nii.gz  -> SMITH

Two subjects with one key. `_group_by_patient` buckets by that key, so
`discover` keeps one of the two scans and hands it BOTH landmark sets to merge
-- silently, in a run that reports success and writes one patient's orientation
under a name that names neither of them.

Every test below is a boundary case of that match: the suffix as a whole token,
as the head of a longer token, as its tail, inside one, repeated, at the start
of a name, as the whole name, with digits after it, and in the other case.

The same rule was fixed first in the shared `sadt_areg_common.pairing`; ASO
carries its own copy of the helper because ASO does not depend on that package.
The two tables differ -- ASO's deliberately has no `_T1`/`_T2`, because there
two timepoints of one subject are two scans to orient -- and
`test_a_timepoint_is_still_not_a_suffix` holds that difference in place.
"""

import pytest

from sadt_aso.cbct import pipeline as cbct_pipeline


stem = cbct_pipeline.patient_stem


# ---------------------------------------------------------------------------
# The suffix as a whole token: still dropped, which is what the table is for.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename,key",
    [
        ("P1_scan.nii.gz", "P1"),
        ("P1_Scan.nii.gz", "P1"),
        ("P1_Or.nii.gz", "P1"),
        ("P1_OR.nii.gz", "P1"),
        ("P1_lm.mrk.json", "P1"),
        ("P1_lm.json", "P1"),
        ("P1_MERGED.nii.gz", "P1"),
        ("P1_Scanreg.nii.gz", "P1"),
        ("P1_Or.gipl.gz", "P1"),
        ("P1_Or.nrrd", "P1"),
    ],
)
def test_a_suffix_that_is_a_whole_token_is_still_dropped(filename, key):
    """The behaviour the table exists for, and the one a fix must not lose."""
    assert stem(filename) == key


def test_a_multi_token_suffix_is_matched_across_its_separator():
    """`_lm_Pred` is two tokens, and has to match as both or as neither."""
    assert stem("P1_lm_Pred.mrk.json") == "P1"
    assert stem("IC_0005_lm_Pred.mrk.json") == "IC_0005"


def test_a_suffix_truncates_rather_than_deletes():
    """Everything the suffix introduces goes with it.

    `P1_Scan_reoriented` is `P1`'s scan, not a patient called
    `P1_reoriented`. Dropping the token alone would invent one.
    """
    assert stem("P1_Scan_reoriented.nii.gz") == "P1"
    assert stem("P1_lm_Or_merged.mrk.json") == "P1"


def test_the_longest_suffix_wins_over_a_shorter_one_inside_it():
    """The table is ordered longest first so `_Scanreg` is not cut at `_Scan`.

    With boundary matching the order no longer decides it on its own -- `_Scan`
    inside `_Scanreg` fails the right-hand boundary -- but both roads have to
    lead to the same place.
    """
    assert stem("P1_Scanreg.nii.gz") == "P1"
    assert stem("P1_lm_Pred.mrk.json") == "P1"


# ---------------------------------------------------------------------------
# The suffix as the HEAD of a longer token. This is the defect.
# ---------------------------------------------------------------------------

HEAD_OF_A_LONGER_TOKEN = [
    ("SMITH_ORTHO_Scan.nii.gz", "SMITH_ORTHO"),
    ("SMITH_ORTHODONTIC_Scan.nii.gz", "SMITH_ORTHODONTIC"),
    ("SMITH_ORLANDO_Scan.nii.gz", "SMITH_ORLANDO"),
    ("P_ORTHO.nii.gz", "P_ORTHO"),
    ("P1_Orion.nii.gz", "P1_Orion"),
    ("P1_Orbit.nii.gz", "P1_Orbit"),
    ("P1_Origin_T1.nii.gz", "P1_Origin_T1"),
    ("P1_scanned.nii.gz", "P1_scanned"),
    ("P1_scanner.nii.gz", "P1_scanner"),
    ("P1_Scanner_T1.nii.gz", "P1_Scanner_T1"),
    ("P1_Scandinavia.nii.gz", "P1_Scandinavia"),
    ("P1_Scanreggae.nii.gz", "P1_Scanreggae"),
    ("P1_lmk.mrk.json", "P1_lmk"),
    ("P1_lms.mrk.json", "P1_lms"),
    ("P1_lmk_Pred.mrk.json", "P1_lmk_Pred"),
    ("P1_MERGEDATA.nii.gz", "P1_MERGEDATA"),
]


@pytest.mark.parametrize("filename,key", HEAD_OF_A_LONGER_TOKEN)
def test_a_suffix_at_the_head_of_a_longer_token_is_not_a_suffix(filename, key):
    """`_OR` opens `_ORTHO`, and ORTHO is part of somebody's file name."""
    assert stem(filename) == key


def test_two_subjects_whose_names_open_with_a_suffix_keep_two_keys():
    """The failure this whole file is about, stated as the thing that broke.

    Both names used to reduce to `SMITH`, so `_group_by_patient` filed all four
    files under one key and `discover` took the first scan in sorted order and
    gave it the other patient's landmarks to merge in.
    """
    first = stem("SMITH_ORTHO_Scan.nii.gz")
    second = stem("SMITH_ORLANDO_Scan.nii.gz")
    assert first != second
    # ... and each still finds its own landmark file.
    assert first == stem("SMITH_ORTHO_lm.mrk.json")
    assert second == stem("SMITH_ORLANDO_lm.mrk.json")


def test_a_scan_and_its_landmarks_still_meet_when_the_name_opens_with_a_suffix():
    """The pairing has to survive the fix, not just the separation.

    `discover` puts a scan and its markups together only if `patient_stem`
    gives them the same key, and the whole semi-automated mode is that pairing.
    """
    assert stem("P1_Orion_scan.nii.gz") == stem("P1_Orion_lm.mrk.json") == "P1_Orion"
    assert stem("P1_lmk_scan.nii.gz") == stem("P1_lmk_lm_Pred.mrk.json") == "P1_lmk"


# ---------------------------------------------------------------------------
# The suffix as the TAIL of, or INSIDE, a longer token.
# ---------------------------------------------------------------------------

def test_a_suffix_at_the_tail_of_a_longer_token_is_not_a_suffix():
    """`PreScan` ends in the word but is one token, so nothing is dropped."""
    assert stem("P1_PreScan.nii.gz") == "P1_PreScan"
    assert stem("P1_SubOr.nii.gz") == "P1_SubOr"


def test_a_suffix_inside_a_longer_token_is_not_a_suffix():
    """Neither edge of the match is a separator."""
    assert stem("P1_PreScanPost.nii.gz") == "P1_PreScanPost"
    assert stem("P1_rescanned.nii.gz") == "P1_rescanned"
    assert stem("P1_MinOrMax.nii.gz") == "P1_MinOrMax"


# ---------------------------------------------------------------------------
# Repeated, at the start, and as the whole name.
# ---------------------------------------------------------------------------

def test_a_repeated_suffix_word_is_cut_at_the_first_aligned_one():
    """`P1_Or1_Or2` is a name; `P1_Or_T1_Or` is a doubly-marked file."""
    assert stem("P1_Or1_Or2.nii.gz") == "P1_Or1_Or2"
    assert stem("P1_Or_T1_Or.nii.gz") == "P1"
    assert stem("P1_scan_scan.nii.gz") == "P1"


def test_a_suffix_word_starting_the_name_is_left_alone():
    """`index > 0` has always guarded this, and it still does.

    A stem that BEGINS with the suffix has no patient in front of it to keep,
    so truncating would leave the empty string and every such file would
    collapse into one nameless patient.
    """
    assert stem("_MERGED.nii.gz") == "_MERGED"
    assert stem("_Or.nii.gz") == "_Or"
    assert stem("_scan.nii.gz") == "_scan"


def test_a_suffix_word_with_no_leading_separator_is_just_a_word():
    """`Orion_T1.nii.gz` has no `_Or` in it at all; the file is patient
    `Orion_T1`."""
    assert stem("Orion_T1.nii.gz") == "Orion_T1"
    assert stem("Scan_only.nii.gz") == "Scan_only"
    assert stem("Or_P1.nii.gz") == "Or_P1"


def test_a_suffix_word_that_is_the_entire_name():
    """Nothing to key on, and the answer must still be a non-empty string."""
    assert stem("Or.nii.gz") == "Or"
    assert stem("scan.nii.gz") == "scan"
    assert stem("MERGED.nii.gz") == "MERGED"


# ---------------------------------------------------------------------------
# Digits after a token, and the separators other than underscore.
# ---------------------------------------------------------------------------

def test_a_digit_after_a_suffix_makes_it_part_of_the_identifier():
    assert stem("P1_Or2.nii.gz") == "P1_Or2"
    assert stem("P1_OR2_T1.nii.gz") == "P1_OR2_T1"
    assert stem("P1_Scan1.nii.gz") == "P1_Scan1"
    assert stem("P1_lm2.mrk.json") == "P1_lm2"


@pytest.mark.parametrize("filename", ["P1_Or-extra.nii.gz", "P1_Or extra.nii.gz"])
def test_every_separator_the_matcher_knows_closes_a_token(filename):
    """A hyphen and a space close a word as plainly as an underscore does, and
    a name a clinician typed can contain either."""
    assert stem(filename) == "P1"


def test_a_dot_inside_a_name_closes_a_token_too():
    """`os.path.splitext` has already taken the extension off, so a dot left in
    the stem is part of the name -- `P1_Or.v2` is `P1`'s second orientation."""
    assert cbct_pipeline._token_aligned_index("P1_Or.v2", "_Or") == 2


def test_a_hyphenated_label_is_not_cut_at_a_word_inside_it():
    """AMASSS writes `CBMASK-Seg`; the `_Or` further along is the real mark."""
    assert stem("P1_CBMASK-Seg_Or.nii.gz") == "P1_CBMASK-Seg"


# ---------------------------------------------------------------------------
# Case, and the timepoints ASO's table deliberately leaves out.
# ---------------------------------------------------------------------------

def test_a_spelling_the_table_does_not_list_is_kept():
    """Suffix matching is case-SENSITIVE, and this fix did not change that.

    `_or` and `_SCAN` are not in `PATIENT_SUFFIXES`, so they survive as part of
    the identity -- the same answer as before the boundary fix. Widening the
    match to any case is a separate decision with its own cost: `_or` is an
    English and a French word.
    """
    assert stem("P1_or.nii.gz") == "P1_or"
    assert stem("P1_SCAN.nii.gz") == "P1_SCAN"
    assert stem("P1_LM.mrk.json") == "P1_LM"


def test_the_two_spellings_the_table_does_list_are_both_dropped():
    assert stem("P1_Or.nii.gz") == stem("P1_OR.nii.gz") == "P1"
    assert stem("P1_Scan.nii.gz") == stem("P1_scan.nii.gz") == "P1"


def test_case_is_preserved_in_what_survives():
    """The key lands in an output path, so a patient keeps the name they got."""
    assert stem("PatIent_A_scan.nii.gz") == "PatIent_A"
    assert stem("p1_Orion_scan.nii.gz") == "p1_Orion"


def test_a_timepoint_is_still_not_a_suffix():
    """ASO's table deliberately has no `_T1`/`_T2`, unlike AREG's.

    Here both timepoints of one subject sit in one folder and are two scans to
    orient; collapsing them dropped the second and merged both landmark sets
    into the survivor. Restated beside the boundary cases because the two
    tables now share a matcher and a later merge must not share the table.
    """
    assert stem("P1_T1.nii.gz") != stem("P1_T2.nii.gz")
    assert stem("P1_T1_Scan.nii.gz") == "P1_T1"
    assert stem("P1_T2_Scan.nii.gz") == "P1_T2"


# ---------------------------------------------------------------------------
# The helper itself, so a failure names the boundary rather than a whole key.
# ---------------------------------------------------------------------------

BOUNDARY_CASES = [
    # stem, suffix, expected index
    ("P1_Or", "_Or", 2),
    ("P1_Orion", "_Or", -1),
    ("P1_Orion_Or", "_Or", 8),
    ("P1_Or_T1", "_Or", 2),
    ("Or_P1", "_Or", -1),
    ("_Or", "_Or", 0),
    ("P1_Or.extra", "_Or", 2),
    ("P1_Or-extra", "_Or", 2),
    ("P1_Or extra", "_Or", 2),
    ("P1_OrExtra", "_Or", -1),
    ("P1_scanned_scan", "_scan", 10),
    ("P1_lmk_lm", "_lm", 6),
    ("IC_0005_lm_Pred", "_lm_Pred", 7),
    ("P1", "_Or", -1),
    ("", "_Or", -1),
]


@pytest.mark.parametrize("text,suffix,index", BOUNDARY_CASES)
def test_the_boundary_helper_reports_the_first_aligned_occurrence(text, suffix, index):
    """Every separator the matcher knows closes a token, not just underscore."""
    assert cbct_pipeline._token_aligned_index(text, suffix) == index


def test_the_helper_checks_the_left_edge_rather_than_assuming_it():
    """Every entry of the table happens to open on `_`, so the left-hand
    boundary is free today. It is checked anyway, so an entry added without one
    is still matched as a token and not as a substring."""
    assert cbct_pipeline._token_aligned_index("P1_Or", "Or") == 3
    assert cbct_pipeline._token_aligned_index("P1_Orion", "Or") == -1
    assert cbct_pipeline._token_aligned_index("Or_P1", "Or") == 0
