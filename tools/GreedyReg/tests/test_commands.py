"""The two greedy invocations, which must stay upstream's verbatim.

`picsl_greedy.Greedy3D.execute(command)` takes the same argument string the
executable does, so upstream's command lines survived the port unchanged. That
is the whole claim the port makes about the registration itself, and the only
way to keep it is to write the flags down here in order.
"""

import pytest

from sadt_greedyreg import pipeline

# The affine search, flag for flag, as `buildRegistrationCommand` emitted it.
UPSTREAM_REGISTRATION = [
    "-d", "3", "-a",
    "-m", "NCC", "4x4x4",
    "-i", "fixed.nii.gz", "moving.nii.gz",
    "-o", "out.mat",
    "-n", "100x100x50x25",
    "-e", "0.5",
    "-search", "100", "10", "20",
    "-dof", "6",
    "-ia", "init.mat",
]

METRIC_FLAGS = {
    "NCC": ["-m", "NCC", "4x4x4"],
    "NMI": ["-m", "NMI"],
    "SSD": ["-m", "SSD"],
}


def registration(metric="NCC", transform_type="Rigid", mask=""):
    return pipeline.registration_command(
        "fixed.nii.gz", "moving.nii.gz", "out.mat", "init.mat", metric, transform_type, mask)


# ---------------------------------------------------------------------------
# Verbatim, in order
# ---------------------------------------------------------------------------

def test_the_registration_command_is_upstreams_flag_for_flag():
    assert registration() == UPSTREAM_REGISTRATION


@pytest.mark.parametrize("metric", list(METRIC_FLAGS))
@pytest.mark.parametrize("transform_type,dof", [("Rigid", "6"), ("Affine", "12")])
def test_every_accepted_combination_builds_the_same_command(metric, transform_type, dof):
    """Six combinations, one shape: only the metric flag and the degrees of
    freedom move, and they move in place."""
    expected = [
        "-d", "3", "-a",
        *METRIC_FLAGS[metric],
        "-i", "fixed.nii.gz", "moving.nii.gz",
        "-o", "out.mat",
        "-n", "100x100x50x25",
        "-e", "0.5",
        "-search", "100", "10", "20",
        "-dof", dof,
        "-ia", "init.mat",
    ]
    assert registration(metric, transform_type) == expected


def test_the_multiresolution_schedule_is_upstreams():
    command = registration()
    assert command[command.index("-n") + 1] == "100x100x50x25"


def test_the_random_search_is_upstreams():
    command = registration()
    assert command[command.index("-search") + 1:command.index("-search") + 4] == ["100", "10", "20"]


def test_the_gradient_step_is_upstreams():
    command = registration()
    assert command[command.index("-e") + 1] == "0.5"


def test_the_fixed_image_comes_before_the_moving_one():
    """`-i fixed moving`, and getting the two the wrong way round is a run that
    succeeds and registers T1 onto T2."""
    command = registration()
    index = command.index("-i")
    assert command[index + 1:index + 3] == ["fixed.nii.gz", "moving.nii.gz"]


def test_no_flag_is_passed_twice():
    command = registration(mask="mask.nii.gz")
    flags = [part for part in command if part.startswith("-") and not part[1:].isdigit()]
    assert len(flags) == len(set(flags))


@pytest.mark.parametrize("metric,expected", sorted(METRIC_FLAGS.items()))
def test_the_metric_flag_carries_a_radius_only_for_ncc(metric, expected):
    assert pipeline.metric_arguments(metric) == expected


@pytest.mark.parametrize("transform_type,dof", [("Rigid", "6"), ("Affine", "12")])
def test_the_degrees_of_freedom_table_is_upstreams(transform_type, dof):
    assert pipeline.DEGREES_OF_FREEDOM[transform_type] == dof


# ---------------------------------------------------------------------------
# The mask
# ---------------------------------------------------------------------------

def test_a_mask_appends_gm_and_changes_nothing_else():
    with_mask = registration(mask="mask.nii.gz")
    assert with_mask == UPSTREAM_REGISTRATION + ["-gm", "mask.nii.gz"]


def test_an_empty_mask_argument_adds_no_flag():
    """`_register_one` passes `mask or ""`, so "no mask" reaches this function
    as the empty string rather than as None."""
    assert "-gm" not in registration(mask="")


# ---------------------------------------------------------------------------
# The resample
# ---------------------------------------------------------------------------

def test_the_resample_command_is_upstreams_flag_for_flag():
    assert pipeline.resample_command("fixed.nii.gz", "moving.nii.gz", "out.nii.gz", "t.mat") == [
        "-d", "3", "-rf", "fixed.nii.gz", "-rm", "moving.nii.gz", "out.nii.gz", "-r", "t.mat"]


def test_the_resample_names_its_output_third_after_rm():
    """`-rm <moving> <output>` is a pair, which is why the stub -- and the
    caller reading a result back -- has to index past the moving image."""
    command = pipeline.resample_command("fixed.nii.gz", "moving.nii.gz", "out.nii.gz", "t.mat")
    assert command[command.index("-rm") + 2] == "out.nii.gz"


def test_the_resample_reads_the_fixed_image_as_the_reference_frame():
    """`-rf` is the frame the moving image lands in: T2 resampled into T1."""
    command = pipeline.resample_command("fixed.nii.gz", "moving.nii.gz", "out.nii.gz", "t.mat")
    assert command[command.index("-rf") + 1] == "fixed.nii.gz"


# ---------------------------------------------------------------------------
# Values this tool does not have
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["ncc", "MI", "", "NCC ", "CC"])
def test_an_unknown_metric_is_refused_by_name(metric):
    """It used to fall through to SSD, so `"ncc"` -- the same word in the wrong
    case, which is exactly what a direct API call sends -- silently changed what
    was optimised while the report still said NCC."""
    with pytest.raises(ValueError) as refusal:
        pipeline.metric_arguments(metric)

    message = str(refusal.value)
    assert repr(metric) in message
    for allowed in pipeline.METRICS:
        assert allowed in message


@pytest.mark.parametrize("transform_type", ["rigid", "affine", "Elastic", ""])
def test_an_unknown_transform_type_is_refused_by_name(transform_type):
    """A bare `KeyError: 'rigid'` from the degrees-of-freedom table said
    nothing a caller could act on; this message is what a 422 carries back."""
    with pytest.raises(ValueError) as refusal:
        registration(transform_type=transform_type)

    message = str(refusal.value)
    assert repr(transform_type) in message
    for allowed in pipeline.TRANSFORMS:
        assert allowed in message


def test_an_unknown_metric_is_refused_when_the_whole_command_is_built():
    with pytest.raises(ValueError, match="Unknown metric"):
        registration(metric="ncc")


def test_check_choices_accepts_every_published_combination():
    for metric in pipeline.METRICS:
        for transform_type in pipeline.TRANSFORMS:
            pipeline.check_choices(metric, transform_type)


def test_check_choices_reports_the_metric_before_the_transform():
    """Both wrong is one message, and it is the first argument's."""
    with pytest.raises(ValueError, match="Unknown metric"):
        pipeline.check_choices("ncc", "rigid")
