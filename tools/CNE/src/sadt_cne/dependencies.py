"""Importing a third-party package late, and saying so when it is not there.

Every heavy or optional import in this tool goes through `require`. Two reasons,
both of them defects upstream had:

* **the schema must cost nothing.** CI imports this package on every pull
  request to publish the schema, and the server regenerates it at startup with
  the tool's own interpreter. Neither may pay for loading an inference engine.
* **a missing reader must cost the notes that need it, not the run.** Upstream
  called `sys.exit(1)` when `pymupdf` OR `python-docx` was absent
  (`CNE_CLI.py:105-115`), so a folder of forty `.txt` notes could not be
  extracted because a PDF reader nobody was going to use was missing.
"""

import importlib


class ToolUnavailableError(Exception):
    """This deployment cannot do it, and no request will change that.

    The server maps an exception by its class NAME -- there is no shared base
    class to inherit, because there is no package shared with the server -- and
    this name answers 503 with the message. That is the right answer for a
    missing dependency: the request was valid, the reason names a package, and
    nothing the caller sends will help.
    """


def require(module_name: str, package_name: str):
    """Import `module_name`, or say which package to install.

    `importlib.import_module` rather than a bare `import` so a test can replace
    it: "a missing reader fails only the notes that need it" is a behaviour
    worth pinning, and it cannot be pinned by uninstalling a dependency.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise ToolUnavailableError(
            f"This server cannot read that note: the '{package_name}' package is not "
            f"installed (import {module_name} failed: {error})."
        ) from error
