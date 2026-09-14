"""What this tool publishes to the server, and the naming rules it has to obey.

`scripts/describe.py` is what turns `run()` into the schema, so these run it and
read the answer rather than restating it -- the same reason the tool's catalogue
is the live registry rather than a copy of it.
"""

import pytest

from sadt_testkit import ToolNotBuilt, is_built, tool_schema

pytestmark = pytest.mark.skipif(
    not is_built("Agent"), reason="run `uv sync` in tools/Agent"
)


@pytest.fixture(scope="module")
def schema():
    try:
        return tool_schema("Agent")
    except ToolNotBuilt as exc:  # pragma: no cover - guarded by the skipif above
        pytest.skip(str(exc))


def test_the_schema_generates(schema):
    """`describe.py` exits 2 on anything it cannot represent, so this is the
    check that the signature is sayable at all."""
    assert schema["name"] == "Agent"
    assert schema["returns"] == "path"


def test_the_first_docstring_line_is_the_description_a_clinician_reads(schema):
    assert schema["description"] == "Route a free-text request to one of this server's tools."


def test_the_tool_declares_a_supervisor(schema):
    """`execute` needs one, and the server has to know before accepting a job
    that something must be injected."""
    assert schema.get("supervisor") is True


def test_no_calls_are_published_and_that_is_the_design(schema):
    """`describe.py` publishes `calls` so the server can refuse at startup a
    tool that chains to one it does not serve. A router's callee is whatever
    the model picks out of the live registry -- the set of names it might call
    IS the set the server serves -- so there is nothing to verify, and the call
    is written through a differently-named parameter for exactly that reason.
    See `execution.py`'s docstring. If this key ever appears, the reasoning
    there has changed and the docstring is stale."""
    assert "calls" not in schema


def test_output_dir_is_required_and_is_the_only_path_the_server_fills(schema):
    assert schema["arguments"]["output_dir"]["type"] == "path"
    assert schema["arguments"]["output_dir"]["required"] is True


def test_the_ollama_model_is_not_published_as_a_hosted_bundle(schema):
    """The server publishes a `path` argument named `model`, `*_model` or
    `*_reference` as a name picked from `DATA/<tool>/models/` and never as
    something a client may upload. An Ollama tag is neither a file nor a bundle
    this server hosts, so it must not be named in a way that reads like one --
    `ASO` shipped `landmark_models` and missed the rule by one letter."""
    names = list(schema["arguments"])
    assert "model_tag" in names
    for name in names:
        assert not (name == "model" or name.endswith("_model")
                    or name.endswith("_reference"))


def test_the_catalogue_is_an_optional_path_a_client_may_upload(schema):
    """Optional, because the live registry is the default source; a path,
    because the server then lets it be uploaded or picked from
    `DATA/Agent/testfiles/`."""
    spec = schema["arguments"]["catalog_file"]
    assert spec["type"] == "path"
    assert spec["required"] is False
    assert spec["default"] == ""


def test_execute_is_a_boolean_that_defaults_to_off(schema):
    """The human approval upstream asked for with a dialog box. It has to be
    visible in the schema, and it has to default to not running anything."""
    spec = schema["arguments"]["execute"]
    assert spec["type"] == "bool"
    assert spec["default"] is False


def test_the_candidate_count_is_published_as_an_argument(schema):
    """Upstream hardcoded 3 in two places. A client can now show all of them."""
    assert schema["arguments"]["candidates"]["type"] == "int"
    assert schema["arguments"]["candidates"]["default"] == 8


def test_folders_is_a_list_so_a_comma_needs_no_escaping(schema):
    assert schema["arguments"]["folders"]["type"] == "list[path]"


def test_the_two_modes_are_published_as_choices(schema):
    assert schema["arguments"]["mode"]["choices"] == [
        "Agent (Automated)", "Ask (Interactive)"
    ]


def test_every_argument_carries_a_description(schema):
    """`describe.py` enforces it, so this documents the property rather than
    discovering it -- and fails loudly if the enforcement is ever relaxed."""
    for name, spec in schema["arguments"].items():
        assert spec.get("description"), name


def test_the_language_model_knobs_are_kept_off_a_clinical_panel(schema):
    for name in ("endpoint", "temperature", "timeout_seconds", "seed", "history"):
        assert schema["arguments"][name].get("hidden") is True, name


def test_the_source_hash_is_published(schema):
    """It is what lets the server cache a schema and know when it is stale."""
    assert len(schema["source_hash"]) == 64
