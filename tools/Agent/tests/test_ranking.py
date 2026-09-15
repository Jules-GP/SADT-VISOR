"""The candidate ranker: what it retrieves, and what it refuses to do.

The two tests that matter most are the last two. Upstream's ranker caught every
exception and silently returned the first three tools in file order while the
router prompt went on saying "choose ONLY from the candidates" -- so with no
network or a stale model cache, a registration request was offered landmarking
and segmentation and nothing else.
"""

import pathlib

import pytest

from sadt_agent import ranking
from sadt_agent.errors import RankingError

from conftest import CATALOG, tool, argument


def names(ranked):
    return [entry["name"] for entry in ranked]


def order(request, tools=None):
    return [entry["name"] for entry, _score in ranking.score_tools(tools or CATALOG, request)]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def test_a_tool_name_becomes_words_a_request_can_match():
    assert "bone" in ranking.tokenize("Bone_Seg")
    assert "seg" in ranking.tokenize("Bone_Seg")


def test_camel_case_in_a_tool_name_is_split():
    tokens = ranking.tokenize("AutoCrop3D")
    assert "auto" in tokens and "crop" in tokens and "3d" in tokens


def test_a_word_glued_to_a_number_matches_either_half():
    tokens = ranking.tokenize("crop3d")
    assert "crop3d" in tokens and "crop" in tokens and "3d" in tokens


def test_stopwords_carry_no_signal():
    tokens = ranking.tokenize("I need the scans of my patient")
    assert "scans" in tokens and "patient" in tokens
    assert not ({"i", "the", "of", "my", "need"} & set(tokens))


def test_a_synonym_expands_at_a_discount():
    terms = ranking.query_terms("align")
    assert terms["align"] == 1.0
    assert terms["register"] == pytest.approx(ranking.SYNONYM_DISCOUNT)


def test_a_literal_match_always_outweighs_its_own_synonym():
    terms = ranking.query_terms("register align")
    assert terms["register"] == 1.0 and terms["align"] == 1.0


def test_a_tool_the_ranker_never_heard_of_is_ranked_on_its_own_words():
    """The property the synonym table must not break: the ranker knows about
    language, never about tools. A tool invented here, with vocabulary that
    appears nowhere in `ranking.py`, is retrieved from its own published text
    with no edit to the module -- which is what the hand-written manifest could
    not do, and why it drifted."""
    invented = list(CATALOG) + [
        tool(
            "Sialography_QC",
            "Score the opacification of a parotid sialogram acquisition.",
            {"acquisitions": argument("path", True, description="The sialograms.")},
        )
    ]
    assert order("score the opacification of my parotid sialograms", invented)[0] \
        == "Sialography_QC"
    assert "sialogra" not in pathlib.Path(ranking.__file__).read_text(
        encoding="utf-8"
    ).lower()


# ---------------------------------------------------------------------------
# Retrieval quality, on the fabricated catalogue
# ---------------------------------------------------------------------------

def test_a_literal_request_reaches_its_tool():
    assert order("segment the bone structures on a CBCT")[0] == "Bone_Seg"


def test_a_paraphrase_with_no_shared_word_still_reaches_its_tool():
    """"line up" shares nothing with "register"; the synonym table is what
    bridges it, and bridging it is the only thing a cross-encoder bought."""
    assert order("line up my second timepoint with the first")[0] == "Timepoint_Reg"


def test_a_request_about_teeth_reaches_the_mesh_tool():
    assert order("number every tooth on my intraoral surface scan")[0] == "Mesh_Label"


def test_a_request_about_written_reports_reaches_the_note_tool():
    assert order("pull the findings out of these clinical reports")[0] == "Note_Reader"


def test_argument_text_is_ranked_too_but_weighs_less_than_the_name():
    """A tool whose ARGUMENT mentions a word must not outrank a tool whose NAME
    is that word."""
    request = "I want a mesh labelled"
    assert order(request)[0] == "Mesh_Label"


def test_ranking_is_deterministic():
    first = order("segment the mandible")
    for _ in range(5):
        assert order("segment the mandible") == first


def test_a_tie_keeps_the_catalogue_order():
    """Two tools with identical text must not swap between runs; a routing that
    changes without the request changing cannot be checked."""
    twins = [
        tool("A_Tool", "identical text", {"x": argument("str")}),
        tool("B_Tool", "identical text", {"x": argument("str")}),
    ]
    assert order("identical text", twins) == ["A_Tool", "B_Tool"]


def test_scores_are_reported_for_every_tool_not_only_the_candidates():
    """`routing.json` names the alternatives with the score that put them there,
    which is only useful if every tool has one."""
    _selected, scores, _narrowed = ranking.select_candidates(CATALOG, "segment", 2)
    assert set(scores) == {entry["name"] for entry in CATALOG}


# ---------------------------------------------------------------------------
# Narrowing -- the upstream defect
# ---------------------------------------------------------------------------

def test_the_candidate_count_is_an_argument_not_a_constant_three():
    """Upstream hardcoded `k=3` at both call sites (`Agent_CLI.py:102,182`), so
    3 of 23 tools were ever shown to the model whatever the request."""
    for limit in (1, 2, 3, 4, 5):
        selected, _scores, _narrowed = ranking.select_candidates(CATALOG, "segment", limit)
        assert len(selected) == limit


def test_zero_candidates_means_every_tool():
    selected, _scores, narrowed = ranking.select_candidates(CATALOG, "segment", 0)
    assert len(selected) == len(CATALOG)
    assert narrowed is False


def test_a_limit_larger_than_the_catalogue_shows_all_of_it():
    selected, _scores, narrowed = ranking.select_candidates(CATALOG, "segment", 99)
    assert len(selected) == len(CATALOG)
    assert narrowed is False


def test_a_ranking_with_no_signal_offers_everything_not_the_first_three():
    """THE defect, inverted. A request sharing nothing with any tool produces
    all-zero scores, so a prefix of that order is the catalogue's file order
    wearing a ranking's clothes. Narrowing is something a ranker earns."""
    selected, scores, narrowed = ranking.select_candidates(
        CATALOG, "zzzz qqqq wwww", 3
    )
    assert narrowed is False
    assert len(selected) == len(CATALOG)
    assert set(scores.values()) == {0.0}


def test_narrowing_is_reported_when_it_happens():
    selected, _scores, narrowed = ranking.select_candidates(
        CATALOG, "segment the bone", 2
    )
    assert narrowed is True and len(selected) == 2


def test_an_empty_catalogue_is_an_error_not_an_empty_candidate_list():
    """A ranker that cannot rank must not narrow, and it must not pretend to
    have ranked. Upstream returned `[]` here and the router was then told to
    choose from nothing."""
    with pytest.raises(RankingError):
        ranking.score_tools([], "anything")


def test_a_broken_document_is_an_error_rather_than_a_silent_fallback():
    """Whatever goes wrong inside the ranker, the answer is never `tools[:k]`.
    Upstream's `except Exception: return scripts[:k]` is the line this asserts
    the absence of."""
    class Exploding(dict):
        def get(self, key, default=None):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        ranking.score_tools([Exploding(name="X")], "anything")


def test_nothing_is_printed_while_ranking(capsys):
    """Upstream's fallback announced itself with `print`, on a stdout the caller
    parsed as JSON -- so the one signal that the candidate set had collapsed was
    also what broke the response."""
    ranking.select_candidates(CATALOG, "segment the bone", 2)
    ranking.select_candidates(CATALOG, "zzzz qqqq", 2)
    assert capsys.readouterr().out == ""
