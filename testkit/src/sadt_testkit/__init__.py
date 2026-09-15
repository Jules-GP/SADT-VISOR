"""Run one tool from another tool's tests, without coupling them.

`tools/ALI` cannot be tested end to end without segmented meshes, and those
come from `tools/Crown_Seg`. Importing crownseg into ali would put the coupling
back that the split exists to remove -- and it could not work anyway, since the
two have different interpreters and irreconcilable dependency sets.

So this runs the other tool the way the SERVER runs it: as a subprocess, through
its own `.venv`, calling its `run()`. Nothing is shared at runtime. This package
is a **development dependency only** and must never appear in a tool's
`[project] dependencies`.

    from sadt_testkit import run_tool

    meshes = run_tool(
        "Crown_Seg",
        meshes=raw_scan,
        model=checkpoint,
        output_dir=tmp_path / "segmented",
    )
    run(scans=meshes, model=..., output_dir=tmp_path / "landmarks")

It has no dependencies of its own, on purpose: a dev dependency that dragged in
a version of anything could perturb the resolution of the tool it is helping to
test, which is the one thing this repository must not let happen.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

__all__ = [
    "ToolFailed",
    "ToolNotBuilt",
    "is_built",
    "repo_root",
    "run_tool",
    "tool_schema",
    "tool_venv_python",
]

_DRIVER = Path(__file__).resolve().parent / "_driver.py"


class ToolNotBuilt(Exception):
    """The tool's venv does not exist yet, and says how to make it."""


class AmbiguousTool(Exception):
    """Two tools answer to that name, so which one would run is a guess.

    Deliberately NOT a ToolNotBuilt, which `is_built` catches: an ambiguous name
    is a mistake in the test, and turning it into "not built" would turn it into
    a silent skip -- the exact failure this resolver was written to end. It has
    to reach whoever wrote the name.
    """


class ToolFailed(Exception):
    """The tool ran and raised. Carries what it printed to stderr."""


def repo_root() -> Path:
    """The checkout holding `tools/`.

    Found by walking up from this file, which works whether the package is
    installed editable (the normal case, `path = "../../testkit"`) or copied.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "tools").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(
        "Could not find the repository root from {}. sadt_testkit expects to be "
        "installed from this checkout.".format(__file__)
    )


def _tool_folder(name: str) -> Path:
    """`tools/<name>`, or the one folder of that name a level down.

    Flat was the only layout when this was written, and the lookup said so.
    Then ALI and AREG were split into engines -- `tools/ALI/ALI_CBCT`,
    `tools/AREG/AREG_IOSCBCT` -- and a nested tool became unreachable by name:
    `is_built("ALI_CBCT")` answered False whatever was built, so every test
    gated on it skipped, quietly and forever.

    That cost something real. `ALI_CBCT`'s whole `test_integration.py` -- five
    tests, including the one checking that the published schema still matches
    `run()`'s signature, which is what the server dispatches against -- had not
    run since the split, and reported itself as "run `uv sync` first": a local
    setup gap, not a dead path.

    One level only, and exactly one match: the layout is `tools/<facade>/<tool>`
    and nothing deeper, and two tools sharing a name would make "which one
    ran" a guess. An explicit `"ALI/ALI_CBCT"` still works -- it is a path and
    is found by the first branch.
    """
    tools = repo_root() / "tools"
    direct = tools / name
    if direct.is_dir():
        return direct

    found = sorted(
        child / name
        for child in tools.iterdir()
        if child.is_dir() and (child / name).is_dir()
    )
    if len(found) == 1:
        return found[0]
    if found:
        raise AmbiguousTool(
            "'{}' names {} tools under tools/: {}. Say which one, as "
            "'<folder>/{}'.".format(
                name, len(found),
                ", ".join(str(path.relative_to(tools)) for path in found), name,
            )
        )
    raise ToolNotBuilt("There is no tool called '{}' under tools/.".format(name))


def tool_venv_python(name: str) -> Path:
    """The interpreter of `tools/<name>/.venv`, nested engines included."""
    root = _tool_folder(name)
    # bin on POSIX, Scripts on Windows -- uv follows the platform.
    for relative in ("bin/python", "Scripts/python.exe"):
        candidate = root / ".venv" / relative
        if candidate.is_file():
            return candidate
    where = root.relative_to(repo_root())
    if not (root / "pyproject.toml").is_file():
        # A GROUPING folder, not a tool: `tools/ALI` and `tools/AREG` hold their
        # engines and are not buildable at all. Saying "run uv sync here" sends a
        # contributor to do something that cannot succeed -- which is what the
        # message did, and what kept a dead gate looking like a local setup gap.
        engines = sorted(
            child.name for child in root.iterdir()
            if child.is_dir() and (child / "pyproject.toml").is_file()
        )
        raise ToolNotBuilt(
            "{} is a folder of tools, not a tool{}.".format(
                where,
                ": name one of " + ", ".join(engines) if engines else "",
            )
        )
    raise ToolNotBuilt(
        "{0} has no .venv. Build it with `cd {0} && uv sync`.".format(where)
    )


def is_built(name: str) -> bool:
    """Whether `run_tool(name, ...)` can work right now.

    Use it to skip an integration test rather than fail it: a contributor
    working on one tool has no reason to have built the others, and CI builds
    each tool in its own job.

        pytest.mark.skipif(not is_built("Crown_Seg"), reason="run uv sync in tools/Crown_Seg")
    """
    try:
        tool_venv_python(name)
    except ToolNotBuilt:
        return False
    return True


def _package_of(name: str) -> str:
    """The one importable package under `tools/<name>/src/`."""
    src = _tool_folder(name) / "src"
    packages = sorted(
        path.name for path in src.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
    )
    if len(packages) != 1:
        raise ToolNotBuilt(
            "expected exactly one package under {}, found {}.".format(
                src, ", ".join(packages) or "none"
            )
        )
    return packages[0]


def _jsonable(value):
    """Paths travel as strings; the driver turns them back."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_tool(name: str, timeout: float = 3600, **params):
    """Call `tools/<name>`'s `run()` in its own venv, and return what it returned.

    A `Path` comes back as a `Path`, a `dict[str, Path]` as a dict of them --
    the same values the server's runner would hand on to the next tool, which
    is the point: an integration test that passes here exercises the real
    handoff rather than a stand-in for it.

    Raises `ToolNotBuilt` when the venv is missing and `ToolFailed`, carrying
    the tool's stderr, when the run itself fails.
    """
    python = tool_venv_python(name)
    package = _package_of(name)
    source = _tool_folder(name) / "src"

    with tempfile.TemporaryDirectory(prefix="sadt_testkit_") as scratch:
        params_file = Path(scratch) / "params.json"
        result_file = Path(scratch) / "result.json"
        params_file.write_text(
            json.dumps({key: _jsonable(value) for key, value in params.items()}),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                str(python), str(_DRIVER),
                "--src", str(source),
                "--package", package,
                "--params", str(params_file),
                "--result", str(result_file),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Not the caller's cwd: a tool must write only under its output_dir,
            # and running it from somewhere neutral is what makes a test that
            # asserts so mean something.
            cwd=scratch,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if completed.returncode != 0:
            raise ToolFailed(
                "{} exited {}.\n--- stderr (last 40 lines) ---\n{}".format(
                    name,
                    completed.returncode,
                    "\n".join(completed.stderr.splitlines()[-40:]),
                )
            )

        result = json.loads(result_file.read_text(encoding="utf-8"))

    if result["kind"] == "path":
        return Path(result["value"])
    return {key: Path(value) for key, value in result["value"].items()}


def tool_schema(name: str, timeout: float = 300) -> dict:
    """The schema `tools/<name>` publishes, as the server would read it.

    The other half of the handoff: `run_tool` checks that one tool's output
    feeds the next, this checks that the next one still declares the argument
    it feeds into. A renamed argument breaks a chain silently otherwise.
    """
    python = tool_venv_python(name)
    describe = repo_root() / "scripts" / "describe.py"
    tool_dir = _tool_folder(name)

    completed = subprocess.run(
        [str(python), str(describe), str(tool_dir)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise ToolFailed(
            "describe.py refused {}:\n{}".format(name, completed.stderr.strip())
        )
    return json.loads(completed.stdout)
