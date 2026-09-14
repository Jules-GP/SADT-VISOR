"""The catalogue of tools this agent may route to, and where it comes from.

**This is the one deliberate design change of the port.** Upstream carried
`Agent_CLI/manifest.yaml`: 891 lines describing 23 tools by hand, with each
tool's parameter list, types, defaults and positional order written a second
time. It had drifted, and a hand-maintained second declaration always does.

Here the catalogue is an INPUT, and its default source is the server's own
registry -- `GET /tools`, which publishes every tool's name, every argument's
type, whether it is required, its choices and its description, generated from
that tool's own `run()` signature with that tool's own interpreter. There is
nothing to keep in step, because there is no second copy.

Two sources, in this order:

1. `catalog`, a JSON file. Explicit, offline, and what every test uses.
2. `SADT_API` + `/tools`, the live registry. `GET /tools` needs no token (the
   server strips `API_TOKEN` from a tool's environment on purpose), and
   `SADT_API` is set by `execution/dispatch.py` for exactly this -- "reaches
   this server from a tool", `server/config.py`. A loopback read of the
   server's own registry is not an outbound call.

If neither is available the run fails naming both. It never guesses.
"""

import json
import os
import re
import urllib.error
import urllib.request

from .errors import CatalogError

# What the server sets so a tool can reach it. Read only when `catalog` is not
# supplied, and only to build `<value>/tools`.
API_ENV = "SADT_API"

# Filled in at dispatch with the job's own output directory, and taken out of
# the published schema before a client ever sees it. A router must never
# propose a value for it: it is not the caller's to choose, and a model that
# invents one proposes writing outside the job.
SERVER_FILLED = ("output_dir",)

# Every type `scripts/describe.py` can emit. Anything else in a catalogue is a
# tool this agent cannot fill parameters for, so it is refused loudly rather
# than routed to with values of an unknown shape.
SCALAR_TYPES = ("path", "str", "int", "float", "bool")
LIST_TYPES = tuple("list[{}]".format(scalar) for scalar in SCALAR_TYPES)
KNOWN_TYPES = SCALAR_TYPES + LIST_TYPES


def load_catalog(catalog_path, timeout_seconds: int = 30):
    """The catalogue and where it came from, as `(tools, source)`.

    `catalog_path` is a path to a JSON file, or an empty value meaning "read the
    live registry".
    """
    if catalog_path:
        path = str(catalog_path)
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise CatalogError(
                "The catalogue at '{}' could not be read: {}.".format(
                    os.path.basename(path), exc
                )
            )
        except ValueError as exc:
            raise CatalogError(
                "The catalogue at '{}' is not valid JSON: {}. It must hold what "
                "GET /tools returns.".format(os.path.basename(path), exc)
            )
        return normalise(payload), "file:{}".format(os.path.basename(path))

    api = os.environ.get(API_ENV, "").strip()
    if not api:
        raise CatalogError(
            "No catalogue. Pass `catalog` -- a JSON file holding what the "
            "server's GET /tools returns -- or run this tool from a server that "
            "sets {}, which is where the live registry is read from.".format(API_ENV)
        )
    return normalise(fetch_catalog(api, timeout_seconds)), "registry:{}".format(api)


def fetch_catalog(api: str, timeout_seconds: int = 30):
    """`GET <api>/tools`, decoded. Stdlib only, and no token: it needs none."""
    url = api.rstrip("/") + "/tools"
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            body = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise CatalogError(
            "The live registry at '{}' could not be read: {}. Pass `catalog` "
            "instead -- a JSON file holding what GET /tools returns.".format(url, exc)
        )
    try:
        return json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise CatalogError("'{}' did not answer with JSON: {}.".format(url, exc))


def normalise(payload):
    """Check a `GET /tools` payload and return it as a list of tool specs.

    Accepts the two shapes that response has been seen in -- a bare list, and
    `{"tools": [...]}` -- and refuses everything else rather than routing
    against a structure it half understands.
    """
    if isinstance(payload, dict):
        for key in ("tools", "scripts"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise CatalogError(
            "A catalogue must be a JSON array of tools, or an object with a "
            "'tools' array, as GET /tools returns. Got {}.".format(
                type(payload).__name__
            )
        )
    if not payload:
        raise CatalogError(
            "The catalogue holds no tools, so there is nothing to route to."
        )

    tools = []
    seen = {}
    for index, entry in enumerate(payload):
        tool = _normalise_tool(entry, index)
        key = normalised_name(tool["name"])
        if key in seen:
            # The same rule the server's registry applies at startup:
            # `Batch_Dental_Seg` and `BatchDentalSeg` are one tool written two
            # ways, and routing to the wrong spelling picks the wrong schema.
            raise CatalogError(
                "The catalogue lists '{}' and '{}', which are the same tool "
                "name written two ways.".format(seen[key], tool["name"])
            )
        seen[key] = tool["name"]
        tools.append(tool)
    return tools


def _normalise_tool(entry, index: int):
    where = "catalogue entry {}".format(index + 1)
    if not isinstance(entry, dict):
        raise CatalogError("{} is not an object.".format(where))

    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise CatalogError("{} has no 'name'.".format(where))
    name = name.strip()

    description = entry.get("description") or ""
    if not isinstance(description, str):
        raise CatalogError("'{}' has a non-string 'description'.".format(name))

    arguments = entry.get("arguments", {})
    if not isinstance(arguments, dict):
        # Upstream's manifest listed parameters as an array of objects each
        # carrying its own `name`, which is what let `p.get("name", "")`
        # collapse every nameless one into a single `""` key -- injected
        # afterwards as a phantom parameter no tool declares. A mapping keyed
        # by name cannot express that shape at all.
        raise CatalogError(
            "'{}' declares 'arguments' as {}. It must be an object keyed by "
            "argument name, as GET /tools publishes it.".format(
                name, type(arguments).__name__
            )
        )

    normalised = {}
    for argument, spec in arguments.items():
        if not isinstance(argument, str) or not argument.strip():
            raise CatalogError(
                "'{}' declares an argument with an empty name. An argument that "
                "cannot be named cannot be sent.".format(name)
            )
        normalised[argument] = _normalise_argument(name, argument, spec)

    return {
        "name": name,
        "description": description.strip(),
        "arguments": normalised,
        "returns": entry.get("returns", ""),
        "supervisor": bool(entry.get("supervisor", False)),
    }


def _normalise_argument(tool: str, argument: str, spec):
    where = "'{}' argument '{}'".format(tool, argument)
    if not isinstance(spec, dict):
        raise CatalogError("{} is not an object.".format(where))

    declared = spec.get("type")
    if not isinstance(declared, str) or not declared:
        raise CatalogError("{} declares no 'type'.".format(where))
    if declared not in KNOWN_TYPES:
        raise CatalogError(
            "{} is of type {!r}, which this agent cannot fill. Known types: "
            "{}.".format(where, declared, ", ".join(KNOWN_TYPES))
        )

    choices = spec.get("choices")
    if choices is not None and not isinstance(choices, list):
        raise CatalogError("{} has a non-list 'choices'.".format(where))

    normalised = {
        "type": declared,
        "required": bool(spec.get("required", False)),
        "description": (spec.get("description") or "").strip(),
    }
    if choices is not None:
        normalised["choices"] = list(choices)
    if "default" in spec:
        normalised["default"] = spec["default"]
    # Presentation, carried through untouched so the prompt can leave a
    # technical argument out of what a clinician is asked about.
    for key in ("hidden", "label", "section"):
        if key in spec:
            normalised[key] = spec[key]
    return normalised


def normalised_name(name: str) -> str:
    """Case- and separator-insensitive tool identity.

    The same comparison the server's registry uses to reject duplicates, so
    `batchdentalseg` from a model resolves to `Batch_Dental_Seg` in the
    catalogue instead of being rejected as a hallucination.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_tool(tools, name: str):
    """The tool a name refers to, or None. Exact first, then normalised."""
    if not isinstance(name, str) or not name.strip():
        return None
    for tool in tools:
        if tool["name"] == name:
            return tool
    key = normalised_name(name)
    for tool in tools:
        if normalised_name(tool["name"]) == key:
            return tool
    return None


def fillable_arguments(tool):
    """The arguments a router may propose values for, in signature order.

    `output_dir` is excluded because the server fills it with the job's own
    output folder; a hidden argument is excluded because it is a technical knob
    the tool's author decided not to put in front of a clinician, and the model
    has no better basis for setting it than the tool's own default does.
    """
    return {
        name: spec
        for name, spec in tool["arguments"].items()
        if name not in SERVER_FILLED and not spec.get("hidden", False)
    }


def missing_required(tool, arguments):
    """Required arguments the caller still has to supply, in signature order."""
    return [
        name
        for name, spec in fillable_arguments(tool).items()
        if spec.get("required", False) and name not in arguments
    ]
