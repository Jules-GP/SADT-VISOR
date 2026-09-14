"""What the model is shown. Half the routing decision is made here.

Upstream truncated a tool's description to 140 characters and its tag list to 8
before the router ever saw it (`Agent_CLI.py:29,31`) -- on a prompt that then
carried a whole manifest's worth of parameter definitions.
"""

from sadt_agent import catalog, prompts

from conftest import CATALOG, tool, argument


TOOLS = catalog.normalise(CATALOG)


def find(name):
    return catalog.find_tool(TOOLS, name)


# ---------------------------------------------------------------------------
# Nothing is truncated
# ---------------------------------------------------------------------------

def test_a_long_description_reaches_the_router_whole():
    long = (
        "Segment craniofacial structures on a cone beam CT scan, one scan or a "
        "whole cohort, writing one labelled volume per structure and optionally "
        "a decimated surface mesh beside it, with the label table recorded in "
        "the run report so a mask stays interpretable."
    )
    assert len(long) > 140
    rendered = prompts.render_tool(tool("Long_Tool", long, {}))
    assert long in rendered


def test_the_description_is_not_cut_at_140_characters():
    long = "x" * 400
    rendered = prompts.render_tool(tool("T", long, {}))
    assert "x" * 400 in rendered


def test_every_candidate_appears_in_the_block():
    block = prompts.candidates_block(TOOLS)
    for entry in TOOLS:
        assert entry["name"] in block


def test_the_router_is_shown_which_arguments_are_required():
    """"Does this tool take two timepoints or one" is what separates three
    registration tools that share a description."""
    rendered = prompts.render_tool(find("Timepoint_Reg"))
    assert "required: baseline, followup" in rendered


def test_the_router_is_not_shown_output_dir():
    block = prompts.candidates_block(TOOLS, catalog.fillable_arguments)
    assert "output_dir" not in block


def test_the_router_is_not_shown_a_hidden_argument():
    block = prompts.candidates_block(TOOLS, catalog.fillable_arguments)
    assert "device" not in block


def test_a_tool_with_no_description_is_still_listed():
    rendered = prompts.render_tool(tool("T", "", {}))
    assert "T" in rendered and "no description" in rendered


# ---------------------------------------------------------------------------
# The router's user message
# ---------------------------------------------------------------------------

def test_the_request_and_the_candidates_both_travel():
    message = prompts.router_user("segment my scans", TOOLS[:2], [])
    assert "segment my scans" in message
    assert TOOLS[0]["name"] in message and TOOLS[1]["name"] in message


def test_folders_are_listed_one_per_line_and_named_as_the_only_paths():
    message = prompts.router_user("go", TOOLS[:1], ["/data/T1", "/data/T2"])
    assert "- /data/T1" in message and "- /data/T2" in message
    assert "do not invent others" in message


def test_no_folders_means_no_folders_section():
    message = prompts.router_user("go", TOOLS[:1], [])
    assert "FOLDERS" not in message


def test_a_folder_with_a_comma_in_its_name_stays_one_line():
    """`Agent_CLI.py:88` did `input.folders.split(",")` with no escaping, so
    `/data/Smith, John/` became two paths that do not exist."""
    message = prompts.folders_block(["/data/Smith, John/T1"])
    assert message.count("- ") == 1
    assert "- /data/Smith, John/T1" in message


# ---------------------------------------------------------------------------
# The extractor's user message
# ---------------------------------------------------------------------------

def test_every_fillable_parameter_is_declared_with_its_type():
    bone = find("Bone_Seg")
    block = prompts.parameter_block(catalog.fillable_arguments(bone))
    assert "- scans (path) [REQUIRED]" in block
    assert "- smoothing (int) [optional] default=5" in block


def test_choices_are_written_out_for_the_model_to_copy():
    bone = find("Bone_Seg")
    block = prompts.parameter_block(catalog.fillable_arguments(bone))
    assert 'CHOICES: "MAND", "MAX", "CB", "UAW"' in block


def test_defaults_are_written_as_json_so_types_survive():
    bone = find("Bone_Seg")
    block = prompts.parameter_block(catalog.fillable_arguments(bone))
    assert "default=false" in block  # not `False`, which is not JSON
    assert 'default=["MAND"]' in block


def test_a_parameter_description_travels():
    bone = find("Bone_Seg")
    block = prompts.parameter_block(catalog.fillable_arguments(bone))
    assert "The CBCT volumes to segment." in block


def test_a_tool_with_no_fillable_parameter_says_so():
    empty = find("No_Argument_Tool")
    assert prompts.parameter_block(catalog.fillable_arguments(empty)) == (
        "(this tool takes no parameters)"
    )


def test_the_extractor_is_told_which_tool_it_is_filling():
    bone = find("Bone_Seg")
    message = prompts.extractor_user(
        "segment", bone, catalog.fillable_arguments(bone), []
    )
    assert message.startswith("TOOL: Bone_Seg")


def test_the_extractor_is_told_not_to_invent_values():
    """A required parameter with no stated value is a question to ask, not a
    path to make up. Making one up produces a run that fails an hour in, on a
    folder that does not exist."""
    assert "Never invent a value" in prompts.EXTRACTOR_SYSTEM


# ---------------------------------------------------------------------------
# The advisor
# ---------------------------------------------------------------------------

def test_the_advisor_is_given_the_candidates_and_told_to_stay_in_them():
    system = prompts.advisor_system(TOOLS)
    assert "Bone_Seg" in system
    assert "say so plainly" in system


def test_the_advisor_is_told_it_runs_nothing():
    assert "you do not run anything" in prompts.ADVISOR_SYSTEM


def test_the_router_is_told_that_no_tool_beats_the_wrong_tool():
    assert "worse than choosing none" in prompts.ROUTER_SYSTEM


def test_a_tool_with_no_arguments_renders_without_an_empty_list():
    rendered = prompts.render_tool(tool("T", "does something", {}))
    assert "required:" not in rendered and "optional:" not in rendered


def test_optional_arguments_are_named_as_optional():
    rendered = prompts.render_tool(
        tool("T", "d", {"a": argument("str", True), "b": argument("str", False)})
    )
    assert "required: a" in rendered and "optional: b" in rendered
