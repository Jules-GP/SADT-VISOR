"""The failures a caller can do something about.

Anything else raised by this tool is a bug and should surface as one. These
replace the server's `base.ToolArgumentError`, which a tool package cannot
import: nothing here knows the server exists. The server maps by exception
class NAME -- `ToolInputError`/`ValueError`/`FileNotFoundError` to 422 with the
message passed through, `ToolUnavailableError` to 503 -- so every message below
is written to be read by whoever sent the request.
"""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""


class ToolUnavailableError(RuntimeError):
    """The tool is installed but the service it needs is not reachable.

    A separate class because the fix is a deployment one, not a request one: no
    argument the caller changes will make an unreachable Ollama endpoint or an
    unpulled model appear. The server answers 503 rather than 422.
    """


class SupervisorRequired(ToolInputError):
    """`execute` was asked for with no way to reach the chosen tool.

    Its own class because nothing about the request is wrong: either the caller
    drops `execute` and runs the proposal themselves, or whatever is running
    this tool has to inject a supervisor -- and a plain `uv run` never will.
    """


class CatalogError(ToolInputError):
    """The catalogue is missing, unreadable, or not shaped like `GET /tools`."""


class RankingError(ToolInputError):
    """The candidate ranker could not produce a candidate set.

    Deliberately fatal. Upstream's ranker caught every exception and fell back
    to `scripts[:k]` -- the first three tools in file order -- while the router
    prompt went on saying "choose ONLY from the candidate list". With no network
    or a stale model cache, a registration request was offered landmarking and
    segmentation and nothing else, and the only sign was a line on stdout that
    also broke the JSON the caller parsed. A ranker that cannot rank must not
    narrow.
    """
