"""Earlier turns of the conversation, so a follow-up message still makes sense.

The history is a JSON array of `{"role", "content"}` -- the shape every chat API
already uses, and the shape the Slicer panel already held in memory
(`Agent/Agent.py:743`, `_appendHistory`).

**It is never recovered by splitting text on emoji.** `OnRetrieveButton`
(`Agent/Agent.py:1126-1131`) reloaded a saved transcript by rewriting every
robot emoji as a person emoji, splitting the whole file on that one character,
and assigning speakers by index parity. A user whose message contained the
emoji -- pasted from anywhere, or written on purpose -- added a fragment, and
every role after it was inverted for the rest of the conversation: the model was
then told that its own answers were the clinician's instructions. Roles here are
data, carried as data.
"""

import json

from .errors import ToolInputError

# What Ollama accepts. `assistant` is the model's own turn; `tool` is not used
# here, and a role outside this set is refused rather than passed through to be
# rejected two layers down with a message about an HTTP body.
ROLES = ("system", "user", "assistant")

# Common spellings of the two speakers, mapped rather than refused: a client
# that stored "agent" or "bot" meant `assistant`, and losing its whole
# conversation over the word is a worse answer than accepting it.
ALIASES = {"agent": "assistant", "bot": "assistant", "ai": "assistant",
           "model": "assistant", "human": "user"}

# A cap, because the history travels in every prompt and an unbounded one
# eventually silently truncates the actual request out of the context window.
# The most recent turns are the ones kept.
MAX_TURNS = 40


def parse_history(raw) -> list:
    """The earlier turns, validated. `""` means none.

    A malformed history is an error, not an empty list. Upstream swallowed
    `json.JSONDecodeError` and continued with `[]` (`Agent_CLI.py:79-84`), so a
    client that got the encoding wrong lost every earlier turn on every message
    and nothing said so -- the model simply kept answering as though each
    message were the first.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return []

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise ToolInputError(
                "`history` is not valid JSON: {}. It must be a JSON array of "
                '{{"role": "user"|"assistant"|"system", "content": "..."}}, or '
                "empty.".format(exc)
            )
    else:
        parsed = raw

    if not isinstance(parsed, list):
        raise ToolInputError(
            "`history` must be a JSON array of turns, not {}.".format(
                type(parsed).__name__
            )
        )

    turns = []
    for index, turn in enumerate(parsed):
        if not isinstance(turn, dict):
            raise ToolInputError(
                "`history` turn {} is not an object.".format(index + 1)
            )
        role = str(turn.get("role", "")).strip().lower()
        role = ALIASES.get(role, role)
        if role not in ROLES:
            raise ToolInputError(
                "`history` turn {} has role {!r}. Use one of: {}.".format(
                    index + 1, turn.get("role"), ", ".join(ROLES)
                )
            )
        content = turn.get("content")
        if not isinstance(content, str):
            raise ToolInputError(
                "`history` turn {} has no string 'content'.".format(index + 1)
            )
        turns.append({"role": role, "content": content})

    return turns[-MAX_TURNS:]
