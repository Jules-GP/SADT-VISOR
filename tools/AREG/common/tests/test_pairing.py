"""The patient key, which is the one thing AREG's engines must agree on."""

from sadt_areg_common import catalogs, pairing

JAW = set(catalogs.JAW_TOKENS)


def test_the_upstream_test_set_pairs_its_two_timepoints():
    """`A2_UpperT1.vtk` and `A2_UpperT2.vtk` are one patient, not two.

    These are upstream's own AREG_test_scans filenames, verbatim -- not a
    renamed version, because the point is that the tool reads the data its own
    project publishes. The jaw and the timepoint run together with no
    separator, so `uppert1` used to match neither the jaw table nor the
    timepoint one, nothing was dropped, and AREG_IOS refused with "no subject
    has a upper arch at both timepoints".
    """
    t1 = pairing.patient_stem("A2_UpperT1.vtk", also_drop=JAW)
    t2 = pairing.patient_stem("A2_UpperT2.vtk", also_drop=JAW)
    assert t1 == t2 == "A2"


def test_an_identifier_that_merely_ends_in_a_timepoint_is_left_alone():
    """`PAT1` is a patient, not a jaw plus a timepoint.

    The split is attempted only when the PREFIX is a known jaw token: `pa` is
    not one, so nothing happens. Widening it to any camelCase-ish boundary
    would start eating patient identifiers, which is the failure this function
    exists to prevent.
    """
    assert pairing.patient_stem("PAT1.vtk", also_drop=JAW) == "PAT1"


def test_the_separated_spelling_still_works():
    """The common case, unchanged: separators do the job on their own."""
    assert pairing.patient_stem("P1_Upper_T1.vtk", also_drop=JAW) == "P1"
    assert pairing.patient_stem("C_0001_T1_Or.nii.gz") == "C_0001"


def test_a_lone_jaw_or_timepoint_token_is_not_split():
    """Nothing to split when the token is already one thing."""
    assert pairing.patient_stem("Lower_gold.vtk", also_drop=JAW) == "gold"
    assert pairing.patient_stem("subject_T0.nii.gz") == "subject"


def test_the_split_only_fires_when_both_halves_are_droppable():
    """A jaw token glued to something that is not a timepoint stays whole."""
    # `upperx` is not jaw+timepoint, so it survives as one token.
    assert pairing.patient_stem("A2_UpperX.vtk", also_drop=JAW) == "A2_UpperX"
