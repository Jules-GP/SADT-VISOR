"""Checking what the model extracted against what the tool actually declares.

Every check here is against a schema generated from the callee's own `run()`
signature, which is the difference the design change bought: a `choices` list is
a real `Literal[...]`, and a type is the real annotation.
"""

import pytest

from sadt_agent import catalog, validation
from sadt_agent.errors import ToolInputError

from conftest import CATALOG


@pytest.fixture
def bone():
    tools = catalog.normalise(CATALOG)
    tool = catalog.find_tool(tools, "Bone_Seg")
    return tool, catalog.fillable_arguments(tool)


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------

def test_a_path_is_kept_as_a_trimmed_string(bone):
    tool, arguments = bone
    values, errors, _ = validation.validate(tool, arguments, {"scans": "  /data/x  "})
    assert values["scans"] == "/data/x" and errors == []


def test_an_empty_path_is_an_error_not_a_value(bone):
    """`Path("")` is the current directory, and truthy. A tool handed one walks
    whatever it happens to be run from."""
    tool, arguments = bone
    _values, errors, _ = validation.validate(tool, arguments, {"scans": "   "})
    assert any("empty string" in error for error in errors)


def test_a_number_written_as_a_string_becomes_a_number(bone):
    tool, arguments = bone
    values, errors, _ = validation.validate(tool, arguments, {"smoothing": "7"})
    assert values["smoothing"] == 7 and isinstance(values["smoothing"], int)
    assert errors == []


def test_a_fractional_value_for_a_whole_number_is_an_error(bone):
    tool, arguments = bone
    _values, errors, _ = validation.validate(tool, arguments, {"smoothing": 2.5})
    assert any("whole number" in error for error in errors)


def test_a_boolean_is_read_from_the_words_a_model_uses(bone):
    tool, arguments = bone
    for spelling, expected in (("true", True), ("Yes", True), ("off", False),
                               ("0", False), (True, True), (0, False)):
        values, errors, _ = validation.validate(
            tool, arguments, {"make_surfaces": spelling}
        )
        assert values["make_surfaces"] is expected, spelling
        assert errors == []


def test_a_word_that_is_not_a_boolean_is_an_error(bone):
    tool, arguments = bone
    _values, errors, _ = validation.validate(tool, arguments, {"make_surfaces": "maybe"})
    assert any("true/false" in error for error in errors)


def test_true_is_never_read_as_the_number_one(bone):
    """`bool` is a subclass of `int` in Python, so an unguarded int check turns
    `make_surfaces: true` into `smoothing: 1`."""
    tool, arguments = bone
    _values, errors, _ = validation.validate(tool, arguments, {"smoothing": True})
    assert any("whole number" in error for error in errors)


def test_a_float_accepts_an_integer_literal():
    tools = catalog.normalise(CATALOG)
    reg = catalog.find_tool(tools, "Timepoint_Reg")
    values, errors, _ = validation.validate(
        reg, catalog.fillable_arguments(reg), {"tolerance": 1}
    )
    assert values["tolerance"] == 1.0 and isinstance(values["tolerance"], float)
    assert errors == []


def test_a_list_stays_a_list(bone):
    tool, arguments = bone
    values, errors, _ = validation.validate(
        tool, arguments, {"structures": ["MAND", "MAX"]}
    )
    assert values["structures"] == ["MAND", "MAX"] and errors == []


def test_a_comma_separated_string_becomes_a_list(bone):
    """Models write lists as strings often enough that refusing them would throw
    away good extractions."""
    tool, arguments = bone
    values, _errors, _ = validation.validate(tool, arguments, {"structures": "MAND, MAX"})
    assert values["structures"] == ["MAND", "MAX"]


def test_a_bracketed_string_becomes_a_list(bone):
    tool, arguments = bone
    values, _errors, _ = validation.validate(
        tool, arguments, {"structures": '["MAND", "CB"]'}
    )
    assert values["structures"] == ["MAND", "CB"]


def test_a_single_value_for_a_list_becomes_a_one_item_list(bone):
    tool, arguments = bone
    values, errors, _ = validation.validate(tool, arguments, {"structures": "MAND"})
    assert values["structures"] == ["MAND"] and errors == []


def test_a_dict_where_a_string_belongs_is_an_error(bone):
    tool, arguments = bone
    _values, errors, _ = validation.validate(tool, arguments, {"scans": {"a": 1}})
    assert errors


def test_null_means_not_stated_rather_than_a_value(bone):
    """There is no nullable type in the contract, so `null` is an omission."""
    tool, arguments = bone
    values, errors, _ = validation.validate(tool, arguments, {"scans": None})
    assert values == {} and errors == []


# ---------------------------------------------------------------------------
# Choices -- real now, because they come from a Literal
# ---------------------------------------------------------------------------

def test_a_value_outside_choices_is_rejected_and_names_the_options(bone):
    """`parameter_validator.py:206` read a `choices` field no manifest entry
    ever set, so this check existed and never once fired. Here it comes from a
    `Literal[...]` in the callee's own signature."""
    tool, arguments = bone
    _values, errors, _ = validation.validate(tool, arguments, {"structures": ["SKULL"]})
    assert any("'MAND'" in error and "SKULL" in error for error in errors)


def test_every_item_of_a_list_argument_is_checked_against_choices(bone):
    """`list[Literal[...]]` is several-of, and one bad option is a 422 for the
    whole request."""
    tool, arguments = bone
    _values, errors, _ = validation.validate(
        tool, arguments, {"structures": ["MAND", "NOPE", "MAX"]}
    )
    assert any("NOPE" in error for error in errors)


def test_a_scalar_choice_is_checked_too():
    tools = catalog.normalise(CATALOG)
    mesh = catalog.find_tool(tools, "Mesh_Label")
    _values, errors, _ = validation.validate(
        mesh, catalog.fillable_arguments(mesh), {"jaw": "Sideways"}
    )
    assert any("Sideways" in error for error in errors)


def test_a_valid_choice_passes(bone):
    tool, arguments = bone
    values, errors, _ = validation.validate(tool, arguments, {"structures": ["CB"]})
    assert values["structures"] == ["CB"] and errors == []


def test_an_argument_with_no_choices_takes_free_text(bone):
    tool, arguments = bone
    values, errors, _ = validation.validate(tool, arguments, {"scans": "/anything/at/all"})
    assert values["scans"] == "/anything/at/all" and errors == []


# ---------------------------------------------------------------------------
# What upstream checked and this does not
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        "/data/cohort.nrrd", "/data/cohort.mha", "/data/cohort.gipl",
        "/data/cohort.dcm", "/data/mesh.vtp", "/data/mesh.obj", "/data/mesh.off",
        "/data/scan.nii.gz", "/data/scan.nii", "/data/mesh.stl", "/data/mesh.vtk",
    ],
)
def test_extensions_the_tools_accept_are_not_rejected_by_a_name_heuristic(value):
    """`parameter_validator.py:191` rejected any argument whose NAME contained
    "folder" or "dir" if its value ended in one of seven extensions -- a list
    missing `.nrrd`, `.mha`, `.gipl`, `.dcm`, `.vtp`, `.obj` and `.off`, every
    one of which these tools read. It was a guess about the argument's meaning
    checked against an incomplete guess about the file's; the schema says
    `path` and does not distinguish a file from a directory, so nothing here
    pretends to."""
    tool = catalog.normalise(
        [{"name": "T", "arguments": {"input_folder": {"type": "path"}}}]
    )[0]
    values, errors, _ = validation.validate(
        tool, catalog.fillable_arguments(tool), {"input_folder": value}
    )
    assert values["input_folder"] == value
    assert errors == []


def test_no_numeric_range_is_invented(bone):
    """`parameter_validator.py:233,242` read `min` and `max`, which no manifest
    entry ever set and which `describe.py` cannot emit. A bound the schema
    cannot express is not a bound this can check."""
    tool, arguments = bone
    values, errors, _ = validation.validate(tool, arguments, {"smoothing": -10_000})
    assert values["smoothing"] == -10_000 and errors == []


def test_no_default_is_invented_for_an_argument_the_user_did_not_mention(bone):
    """`complete_with_defaults` copied every optional default into the proposal.
    The defaults are in the callee's own signature; restating them adds a second
    copy that can only drift, and hides which values the USER actually asked
    for."""
    tool, arguments = bone
    values, _errors, _ = validation.validate(tool, arguments, {"scans": "/data"})
    assert values == {"scans": "/data"}


# ---------------------------------------------------------------------------
# Unknown arguments
# ---------------------------------------------------------------------------

def test_an_argument_the_tool_does_not_take_is_dropped_and_reported(bone):
    """Forwarding it moves the 422 to a place with less context."""
    tool, arguments = bone
    values, errors, unknown = validation.validate(
        tool, arguments, {"scans": "/data", "sharpen": True}
    )
    assert values == {"scans": "/data"}
    assert unknown == ["sharpen"]
    assert any("sharpen" in error for error in errors)


def test_output_dir_proposed_by_the_model_is_dropped(bone):
    """It is not in the fillable set, so a model that proposes one is proposing
    a write outside the job -- and it never reaches the callee."""
    tool, arguments = bone
    values, _errors, unknown = validation.validate(
        tool, arguments, {"output_dir": "/etc"}
    )
    assert values == {} and unknown == ["output_dir"]


def test_a_hidden_argument_proposed_by_the_model_is_dropped(bone):
    tool, arguments = bone
    values, _errors, unknown = validation.validate(tool, arguments, {"device": "cpu"})
    assert values == {} and unknown == ["device"]


def test_an_extracted_field_that_is_not_an_object_is_reported_not_raised(bone):
    """A badly shaped answer is a fact about this proposal, not a failed run."""
    tool, arguments = bone
    values, errors, unknown = validation.validate(tool, arguments, "MAND")
    assert values == {} and unknown == [] and errors


def test_coerce_raises_for_an_unknown_type():
    with pytest.raises(ToolInputError):
        validation.coerce("x", {"type": "complex"}, 1)
