"""AREG_IOSCBCT's argument rules and its published schema: no GPU, no weights,
no network.

Split out of the single `tools/AREG/tests/test_run.py` AREG had before it
became three tools; see AREG_CBCT/tests/test_run.py for why none of them ran.

The tools this one drives are stood in for by a fake supervisor, which is all a
tool can see of them: five members, duck-typed, nothing imported across venvs.
`FakeSup` and the cohort builders live in `conftest.py`.

Every rule below is checked BEFORE a file is read -- most of these tests pass
paths that do not exist, and reaching the file system would raise something
else entirely. That is the point: a request that cannot work has to come back
in a second, not after an hour of registration.
"""

import inspect
import os
import typing

import pytest

from conftest import FakeSup, cohort, run_registration
from sadt_areg_ioscbct import dispatch, run, tools
from sadt_areg_ioscbct.layout import LAYOUT
from sadt_areg_common import catalogs
from sadt_areg_common.errors import SupervisorRequired, ToolInputError


def _choices(argument):
    """The `Literal` options `run()` publishes for one argument."""
    hint = typing.get_type_hints(run)[argument]
    if typing.get_origin(hint) is list:
        hint = typing.get_args(hint)[0]
    return list(typing.get_args(hint))


def _main(tmp_path=None, **overrides):
    arguments = {
        "ios": "/nonexistent/ios",
        "cbct": "/nonexistent/cbct",
        "output_dir": str(tmp_path / "out") if tmp_path else "/nonexistent/out",
        "automation": catalogs.AUTOMATION_REGISTRATION,
    }
    arguments.update(overrides)
    return dispatch.main(**arguments)


def test_every_tool_is_named_by_string():
    """`sup.run("ASO", ...)`, never `sup.ASO(...)`. A typo in a string is
    greppable and tools.py is the whole call graph; a typo in an attribute is an
    AttributeError an hour into a job.

    Here rather than in AREG_CBCT, which was where the single pre-split suite
    left it: this is the tool that drives all four, so it is the only one whose
    tools.py can be expected to name all four.
    """
    source = open(tools.__file__, encoding="utf-8").read()
    assert 'sup.run("' in source
    for tool in ("Crown_Seg", "ALI_CBCT", "ALI_IOS", "ASO"):
        assert f'"{tool}"' in source, tool


# ---------------------------------------------------------------------------
# The modes
# ---------------------------------------------------------------------------

def test_the_default_mode_is_the_one_that_needs_no_other_tool():
    """A default that cannot run without a supervisor and three model bundles
    reads as a broken tool rather than as a default."""
    assert inspect.signature(run).parameters["automation"].default == (
        catalogs.AUTOMATION_REGISTRATION
    )


def test_the_published_modes_are_the_catalog_s_own_for_this_modality():
    """`Literal` takes literals only, so the signature is a second declaration
    of the same set. A mode offered here and refused by `main` is a picker that
    produces 422s; one `main` accepts and the signature omits is a mode no
    client can reach.

    The ORDER differs on purpose -- the signature leads with the default, so
    a client's picker opens on the mode that needs nothing."""
    assert set(_choices("automation")) == set(
        catalogs.AUTOMATION_BY_MODALITY[catalogs.MODALITY_IOSCBCT]
    )
    assert _choices("automation")[0] == catalogs.AUTOMATION_REGISTRATION


def test_the_oriented_mode_belongs_to_the_cbct_tool_alone(tmp_path):
    """Orienting before registering is a CBCT step, and there is nothing to
    orient an intraoral scan onto in this mode."""
    assert catalogs.AUTOMATION_ORIENTED not in _choices("automation")
    with pytest.raises(ToolInputError) as raised:
        _main(tmp_path, automation=catalogs.AUTOMATION_ORIENTED)
    assert catalogs.AUTOMATION_ORIENTED in str(raised.value)


def test_an_unknown_mode_names_the_value_and_what_is_offered(tmp_path):
    """`Literal` is published, not enforced -- the runner calls `run(**params)`
    from a JSON object -- so a stale client has to be told."""
    with pytest.raises(ToolInputError) as raised:
        _main(tmp_path, automation="Magic")
    message = str(raised.value)
    assert "'Magic'" in message
    for mode in catalogs.AUTOMATION_BY_MODALITY[catalogs.MODALITY_IOSCBCT]:
        assert mode in message


def test_an_omitted_mode_falls_back_to_registration(tmp_path):
    """`main` is also called directly by tests and by another tool, where the
    signature's default does not apply."""
    with pytest.raises(ToolInputError, match="Registration mode takes"):
        _main(tmp_path, automation=None)


# ---------------------------------------------------------------------------
# What each mode needs
# ---------------------------------------------------------------------------

def test_registration_mode_without_landmarks_names_both_arguments(tmp_path):
    with pytest.raises(ToolInputError) as raised:
        _main(tmp_path)
    message = str(raised.value)
    assert "ios_landmarks" in message and "cbct_landmarks" in message


def test_registration_mode_with_only_one_side_is_still_refused(tmp_path):
    with pytest.raises(ToolInputError, match="Registration mode takes"):
        _main(tmp_path, ios_landmarks="/nonexistent/lm")
    with pytest.raises(ToolInputError, match="Registration mode takes"):
        _main(tmp_path, cbct_landmarks="/nonexistent/lm")


def test_registration_mode_needs_no_supervisor_at_all(tmp_path):
    """This is what makes the tool usable standalone: the mode that predicts
    nothing calls nothing."""
    cohort(tmp_path)
    assert run_registration(tmp_path, sup=None)


@pytest.mark.parametrize(
    "automation", [catalogs.AUTOMATION_SEMI, catalogs.AUTOMATION_FULLY]
)
def test_an_automated_mode_without_a_supervisor_refuses_at_the_door(tmp_path, automation):
    """Checked up front, before a single file is read. Crown_Seg is the first
    of the three, so it is the one named."""
    with pytest.raises(SupervisorRequired) as raised:
        _main(tmp_path, automation=automation)
    assert "Crown_Seg" in str(raised.value)
    assert automation in str(raised.value)


def test_the_rules_run_before_anything_is_read():
    """Every case above passes paths that do not exist."""
    assert not os.path.exists("/nonexistent/ios")


def test_an_unreadable_input_is_reported_rather_than_crashing_the_batch(tmp_path):
    """`discover` walks a folder that is not there without complaining, so the
    refusal is the pairing one -- which names what it was looking for."""
    cohort(tmp_path)
    with pytest.raises(ToolInputError, match="No patient has both"):
        run_registration(tmp_path, cbct="/nonexistent/cbct")


# ---------------------------------------------------------------------------
# max_dist
# ---------------------------------------------------------------------------

def _captured_max_dist(tmp_path, monkeypatch, **overrides):
    from sadt_areg_ioscbct import pipeline

    seen = []
    real = pipeline.register_one

    def spy(*args, **kwargs):
        seen.append(kwargs["max_dist"])
        return real(*args, **kwargs)

    monkeypatch.setattr(dispatch.pipeline, "register_one", spy)
    cohort(tmp_path)
    run_registration(tmp_path, **overrides)
    return seen


def test_a_max_dist_of_zero_means_upstreams_own_value(tmp_path, monkeypatch):
    """0 is what an untouched numeric field sends, and it is not a distance
    anybody meant -- every point would be further from its neighbour than
    that, and the ICP would match nothing."""
    assert _captured_max_dist(tmp_path, monkeypatch, max_dist=0.0) == [1.5]


def test_an_omitted_max_dist_means_the_same(tmp_path, monkeypatch):
    assert _captured_max_dist(tmp_path, monkeypatch) == [1.5]


@pytest.mark.parametrize("given", [0.25, 1.5, 12.0])
def test_a_max_dist_the_caller_named_is_the_one_used(tmp_path, monkeypatch, given):
    assert _captured_max_dist(tmp_path, monkeypatch, max_dist=given) == [given]


def test_an_integer_max_dist_is_accepted_as_a_distance(tmp_path, monkeypatch):
    """A form field sends `3`, not `3.0`, and the ICP compares it against
    floats."""
    assert _captured_max_dist(tmp_path, monkeypatch, max_dist=3) == [3.0]


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def test_the_layout_only_names_arguments_the_signature_offers():
    """`describe.py` refuses a hint naming an argument `run()` does not take,
    so a stale entry here is a tool that will not publish its schema at all."""
    assert set(LAYOUT) <= set(inspect.signature(run).parameters)


def test_every_layout_condition_names_a_mode_this_tool_has():
    """A `visible_when` naming a mode that does not exist hides the field for
    good, and a client cannot tell that from a field the tool never
    declared."""
    offered = set(catalogs.AUTOMATION_BY_MODALITY[catalogs.MODALITY_IOSCBCT])
    for argument, hints in LAYOUT.items():
        condition = hints.get("visible_when")
        if not condition:
            continue
        assert set(condition) <= set(inspect.signature(run).parameters), argument
        wanted = condition["automation"]
        for mode in (wanted if isinstance(wanted, list) else [wanted]):
            assert mode in offered, (argument, mode)


def test_the_landmark_folders_are_hidden_in_the_modes_that_overwrite_them():
    """Showing them in a mode that predicts its own is how a user comes to
    believe their files were used."""
    for argument in ("ios_landmarks", "cbct_landmarks"):
        assert LAYOUT[argument]["visible_when"] == {
            "automation": catalogs.AUTOMATION_REGISTRATION
        }


def test_the_orientation_reference_is_shown_only_in_the_mode_that_orients():
    assert LAYOUT["cbct_reference"]["visible_when"] == {
        "automation": catalogs.AUTOMATION_FULLY
    }


def test_every_published_argument_has_a_label(tmp_path):
    """The client builds its panel from these; an argument with none falls back
    to its identifier, which is how a panel ends up saying `cbct_landmarks`
    above a file picker."""
    published = [
        name for name in inspect.signature(run).parameters
        if name not in ("output_dir", "sup")
    ]
    assert sorted(LAYOUT) == sorted(published)
    for name in published:
        assert LAYOUT[name].get("label"), name


def test_the_two_inputs_are_named_for_the_modality_not_for_a_timepoint():
    """NOT longitudinal, unlike its two siblings: one timepoint imaged two
    ways, which is why the arguments are `ios` and `cbct` rather than `t1` and
    `t2`."""
    parameters = set(inspect.signature(run).parameters)
    assert {"ios", "cbct"} <= parameters
    assert not {"t1", "t2"} & parameters


def test_the_supervisor_is_keyword_only():
    """It is injected by the runner, never sent by a client, and
    `describe.py` publishes only what precedes the bare `*`."""
    signature = inspect.signature(run)
    assert signature.parameters["sup"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["sup"].default is None


def test_a_supervisor_is_accepted_and_unused_in_registration_mode(tmp_path):
    """A server always supplies one. The mode that needs nothing must not start
    calling something because it was given the means to."""
    cohort(tmp_path)
    sup = FakeSup(tmp_path)
    run_registration(tmp_path, sup=sup)
    assert sup.calls == []
