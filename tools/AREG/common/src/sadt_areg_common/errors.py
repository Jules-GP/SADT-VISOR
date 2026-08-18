"""The failures a caller can do something about.

Anything else raised by this tool is a bug and should surface as one. These
replace the server's `base.ToolInputError` / `base.ToolUnavailableError`,
which a tool package cannot import: nothing here knows the server exists.
"""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""


class ToolUnavailableError(RuntimeError):
    """An engine's stack is not installed in this venv.

    Real for AREG twice over: the IOS engine needs pytorch3d, compiled from
    source, and the CBCT engine needs `itk-elastix` on top of plain itk. A venv
    where `uv sync` succeeded can still be one where either half cannot start,
    so this is raised at the point of use rather than at import.
    """


class SupervisorRequired(ToolInputError):
    """A mode was asked for that needs another tool, with no way to reach it.

    Its own class because the fix is unlike any other input error: nothing about
    the request is wrong. Either the caller supplies what the other tool would
    have produced, or whatever runs this has to inject a supervisor -- and a
    plain `uv run` never will, which is the standalone case the README documents.
    """
