"""Earlier turns of the conversation, and the one way they must never be read.

`OnRetrieveButton` (`Agent/Agent.py:1126-1131`) recovered a saved transcript by
rewriting every robot emoji as a person emoji, splitting the file on that one
character, and assigning speakers by index parity. Roles here are data.
"""

import json

import pytest

from sadt_agent.conversation import MAX_TURNS, parse_history
from sadt_agent.errors import ToolInputError


def test_an_empty_history_is_no_turns():
    assert parse_history("") == []
    assert parse_history("   ") == []
    assert parse_history(None) == []
    assert parse_history("[]") == []


def test_turns_survive_in_order_with_their_roles():
    raw = json.dumps([
        {"role": "user", "content": "segment my CBCTs"},
        {"role": "assistant", "content": "which structures?"},
        {"role": "user", "content": "mandible"},
    ])
    assert parse_history(raw) == [
        {"role": "user", "content": "segment my CBCTs"},
        {"role": "assistant", "content": "which structures?"},
        {"role": "user", "content": "mandible"},
    ]


def test_an_emoji_in_a_message_does_not_reassign_a_single_role():
    """THE defect. Upstream split the transcript on the robot emoji and assigned
    speakers by index parity, so a user who pasted or typed one added a fragment
    and inverted every role after it -- the model was then told its own answers
    were the clinician's instructions."""
    raw = json.dumps([
        {"role": "user", "content": "here is a robot 🤖 and a person 👨"},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": "🤖🤖🤖 now segment it"},
    ])
    turns = parse_history(raw)
    assert [turn["role"] for turn in turns] == ["user", "assistant", "user"]
    assert turns[0]["content"] == "here is a robot 🤖 and a person 👨"
    assert turns[2]["content"] == "🤖🤖🤖 now segment it"


def test_a_message_that_is_nothing_but_emoji_is_still_one_turn():
    turns = parse_history(json.dumps([{"role": "user", "content": "👨"}]))
    assert turns == [{"role": "user", "content": "👨"}]


def test_a_malformed_history_is_an_error_not_a_forgotten_conversation():
    """Upstream swallowed `json.JSONDecodeError` and continued with `[]`
    (`Agent_CLI.py:79-84`), so a client that got the encoding wrong lost every
    earlier turn on every message and nothing said so."""
    with pytest.raises(ToolInputError) as raised:
        parse_history("not json at all")
    assert "not valid JSON" in str(raised.value)


def test_a_history_that_is_not_an_array_says_what_it_got():
    with pytest.raises(ToolInputError) as raised:
        parse_history('{"role": "user", "content": "hi"}')
    assert "array of turns" in str(raised.value)


def test_a_turn_that_is_not_an_object_names_its_position():
    with pytest.raises(ToolInputError) as raised:
        parse_history('[{"role": "user", "content": "a"}, "oops"]')
    assert "turn 2" in str(raised.value)


def test_an_unknown_role_is_refused_and_lists_the_known_ones():
    with pytest.raises(ToolInputError) as raised:
        parse_history('[{"role": "narrator", "content": "a"}]')
    assert "narrator" in str(raised.value) and "assistant" in str(raised.value)


def test_the_common_spellings_of_the_two_speakers_are_accepted():
    """A client that stored "agent" meant `assistant`; losing the whole
    conversation over the word is a worse answer than accepting it."""
    raw = json.dumps([
        {"role": "Agent", "content": "a"},
        {"role": "bot", "content": "b"},
        {"role": "human", "content": "c"},
    ])
    assert [turn["role"] for turn in parse_history(raw)] == [
        "assistant", "assistant", "user"
    ]


def test_a_turn_with_no_string_content_is_refused():
    with pytest.raises(ToolInputError) as raised:
        parse_history('[{"role": "user", "content": 42}]')
    assert "content" in str(raised.value)


def test_an_already_parsed_list_is_accepted():
    """So a caller inside Python does not have to serialize what it just built."""
    turns = parse_history([{"role": "user", "content": "hi"}])
    assert turns == [{"role": "user", "content": "hi"}]


def test_a_long_history_keeps_the_most_recent_turns():
    """The history travels in every prompt; unbounded, it eventually truncates
    the actual request out of the context window."""
    raw = [{"role": "user", "content": str(index)} for index in range(MAX_TURNS + 10)]
    turns = parse_history(json.dumps(raw))
    assert len(turns) == MAX_TURNS
    assert turns[-1]["content"] == str(MAX_TURNS + 9)
