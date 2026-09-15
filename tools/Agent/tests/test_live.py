"""The only tests that talk to a real language model.

Deselected by default (`addopts = -m 'not gpu and not models'`), because a green
line that depends on an Ollama server being up with a 5 GB model pulled says
nothing about the code. Run them by hand and report the result in the PR:

    OLLAMA_HOST=http://127.0.0.1:11434 uv run pytest -m models -o addopts=

They are the check the stubbed suite structurally cannot make: that the prompts
this tool builds actually make a model answer with a name from the list.
"""

import json
import urllib.request

import pytest

from sadt_agent import DECISION_NAME, MODE_ASK, llm, run

from conftest import CATALOG

# Captured before conftest's `no_network` fixture replaces it.
_REAL_URLOPEN = urllib.request.urlopen

pytestmark = pytest.mark.models


@pytest.fixture(autouse=True)
def allow_network(no_network, monkeypatch):
    """Undo the suite-wide network ban, for these tests only."""
    monkeypatch.setattr("urllib.request.urlopen", _REAL_URLOPEN)


@pytest.fixture(scope="session")
def endpoint():
    resolved = llm.resolve_endpoint("")
    try:
        with _REAL_URLOPEN(resolved + "/api/tags", timeout=5) as response:
            tags = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - any failure means "not available"
        pytest.skip("no Ollama at {}: {}".format(resolved, exc))
    names = {entry.get("name") for entry in tags.get("models", [])}
    if "qwen3:8b" not in names:
        pytest.skip("qwen3:8b is not pulled on {}".format(resolved))
    return resolved


@pytest.fixture
def catalog_path(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(CATALOG), encoding="utf-8")
    return path


def decision(output_dir):
    return json.loads((output_dir / DECISION_NAME).read_text(encoding="utf-8"))


def test_a_segmentation_request_routes_to_the_segmentation_tool(
    tmp_path, catalog_path, endpoint
):
    output_dir = run(
        prompt="I have 40 cone beam CTs and I need the mandible segmented out of each",
        output_dir=tmp_path / "out",
        catalog_file=catalog_path,
        endpoint=endpoint,
    )
    assert decision(output_dir)["tool"] == "Bone_Seg"


def test_a_paraphrased_registration_request_routes_to_the_registration_tool(
    tmp_path, catalog_path, endpoint
):
    output_dir = run(
        prompt="line up the follow-up scan with the baseline one so I can compare them",
        output_dir=tmp_path / "out",
        catalog_file=catalog_path,
        endpoint=endpoint,
    )
    assert decision(output_dir)["tool"] == "Timepoint_Reg"


def test_a_folder_the_user_gave_is_extracted_verbatim(
    tmp_path, catalog_path, endpoint
):
    output_dir = run(
        prompt="segment the bone in /data/cohort_A using the model at /models/bone_v3",
        output_dir=tmp_path / "out",
        catalog_file=catalog_path,
        folders=["/data/cohort_A", "/models/bone_v3"],
        endpoint=endpoint,
    )
    made = decision(output_dir)
    assert made["tool"] == "Bone_Seg"
    assert made["arguments"].get("scans") == "/data/cohort_A"


def test_the_same_request_routes_the_same_way_twice(
    tmp_path, catalog_path, endpoint
):
    """Temperature 0 and a fixed seed. A routing that changes without the
    request changing cannot be checked by anyone."""
    first = decision(run(
        prompt="number every tooth on my intraoral scans",
        output_dir=tmp_path / "a", catalog_file=catalog_path, endpoint=endpoint,
    ))
    second = decision(run(
        prompt="number every tooth on my intraoral scans",
        output_dir=tmp_path / "b", catalog_file=catalog_path, endpoint=endpoint,
    ))
    assert first["tool"] == second["tool"] == "Mesh_Label"


def test_a_request_nothing_can_do_is_refused_rather_than_routed(
    tmp_path, catalog_path, endpoint
):
    """The router is told that choosing the wrong tool is worse than choosing
    none, because these tools run for minutes to hours on patient data."""
    output_dir = run(
        prompt="book me a flight to Lisbon on Tuesday",
        output_dir=tmp_path / "out",
        catalog_file=catalog_path,
        endpoint=endpoint,
    )
    made = decision(output_dir)
    assert made["tool"] is None or made["confidence"] < 0.5


def test_ask_mode_answers_with_advice_naming_real_tools(
    tmp_path, catalog_path, endpoint
):
    output_dir = run(
        prompt="what order should I run things in to compare two timepoints?",
        output_dir=tmp_path / "out",
        catalog_file=catalog_path,
        mode=MODE_ASK,
        endpoint=endpoint,
    )
    advice = (output_dir / "advice.md").read_text(encoding="utf-8")
    assert advice.strip()
    assert any(entry["name"] in advice for entry in CATALOG)


def test_the_whole_catalogue_can_be_shown_at_once(tmp_path, catalog_path, endpoint):
    """`candidates=0`. Upstream could only ever show three."""
    output_dir = run(
        prompt="pull the findings out of these clinical reports",
        output_dir=tmp_path / "out",
        catalog_file=catalog_path,
        candidates=0,
        endpoint=endpoint,
    )
    assert decision(output_dir)["tool"] == "Note_Reader"
