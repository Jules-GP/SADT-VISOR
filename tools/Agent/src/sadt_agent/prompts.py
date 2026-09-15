"""Every prompt this tool sends, built from the live catalogue.

Nothing here is truncated. Upstream cut each tool's description to 140
characters and its tag list to 8 before the router ever saw it
(`Agent_CLI.py:29,31`), mid-sentence and mid-word, on a prompt that then went on
to carry an entire manifest's worth of parameter definitions. Sixteen tools'
full one-line descriptions are about 1200 characters; there is nothing to save.
"""

import json

# The router's contract with the model, and the router's own post-check. The
# model is told to pick from the candidates; `routing.py` then verifies that it
# did, because an instruction is not a guarantee.
ROUTER_SYSTEM = """\
You are a tool router for a medical imaging server. Answer with ONE JSON object \
and nothing else.

Schema:
{"tool": string|null, "confidence": number between 0 and 1, "reason": string}

Rules:
- Choose `tool` ONLY from the CANDIDATES list in the user message, spelled \
exactly as it is written there.
- If no candidate does what the user asked, set tool to null and confidence to \
0.2 or less. Choosing the wrong tool is worse than choosing none: these tools \
run for minutes to hours on patient data.
- Do not invent tool names and do not answer with a tool that is not listed.
- `reason` is one sentence saying why, naming what in the request decided it.\
"""

EXTRACTOR_SYSTEM = """\
You extract a tool's parameters from a user's request. Answer with ONE JSON \
object and nothing else.

Schema:
{"extracted": {parameter name: value}, "confidence": number between 0 and 1, \
"notes": string}

Rules:
- Extract ONLY parameters the user actually stated or that follow directly from \
the folders listed. Never invent a value, never guess a path, and never fill a \
parameter just because it is required -- a missing required parameter is a \
question to ask, not a value to make up.
- Leave out any parameter the user did not mention. Its declared default \
applies.
- Types: booleans as true/false, numbers unquoted, lists as JSON arrays, \
everything else as a JSON string.
- A parameter with CHOICES must take one of those choices, copied exactly.\
"""

ADVISOR_SYSTEM = """\
You are an imaging methodology consultant for a dental and craniofacial \
research group. You advise; you do not run anything.

Answer in short Markdown. Recommend tools only from the AVAILABLE TOOLS list \
below, naming each one exactly as it is written there, and say in what order \
they should be run and why. Name the parameters that matter and say what has to \
be prepared before each step. If nothing available does what was asked, say so \
plainly instead of recommending the nearest thing.

AVAILABLE TOOLS
{tools}\
"""


def render_tool(tool, arguments=None) -> str:
    """One tool as the router sees it: name, description, and its arguments.

    The argument list is what makes a routing decision possible at all -- "does
    this tool take two timepoints or one" separates three registration tools
    that share a description.
    """
    lines = ["- {}: {}".format(tool["name"], tool.get("description") or "(no description)")]
    if arguments is None:
        arguments = tool.get("arguments") or {}
    required = [name for name, spec in arguments.items() if spec.get("required")]
    optional = [name for name, spec in arguments.items() if not spec.get("required")]
    if required:
        lines.append("    required: " + ", ".join(required))
    if optional:
        lines.append("    optional: " + ", ".join(optional))
    return "\n".join(lines)


def candidates_block(candidates, arguments_for=None) -> str:
    """The CANDIDATES section of the router prompt."""
    return "\n".join(
        render_tool(tool, None if arguments_for is None else arguments_for(tool))
        for tool in candidates
    )


def router_user(prompt: str, candidates, folders, arguments_for=None) -> str:
    parts = ["USER REQUEST:\n" + prompt.strip()]
    if folders:
        parts.append(folders_block(folders))
    parts.append(
        "CANDIDATES (choose exactly one name from this list, or null):\n"
        + candidates_block(candidates, arguments_for)
    )
    return "\n\n".join(parts)


def folders_block(folders) -> str:
    """The paths the caller says are relevant, as the source of truth for paths."""
    listed = "\n".join("- " + str(folder) for folder in folders)
    return (
        "FOLDERS THE USER HAS PROVIDED (the only paths that exist; use them "
        "verbatim, do not invent others):\n" + listed
    )


def parameter_block(arguments) -> str:
    """Every fillable parameter of the chosen tool, in signature order."""
    lines = []
    for name, spec in arguments.items():
        head = "- {} ({}) [{}]".format(
            name, spec["type"], "REQUIRED" if spec.get("required") else "optional"
        )
        if "default" in spec:
            head += " default={}".format(_literal(spec["default"]))
        if spec.get("choices"):
            head += " CHOICES: {}".format(
                ", ".join(_literal(choice) for choice in spec["choices"])
            )
        description = spec.get("description")
        lines.append(head + (": " + description if description else ""))
    return "\n".join(lines) or "(this tool takes no parameters)"


def extractor_user(prompt: str, tool, arguments, folders) -> str:
    parts = [
        "TOOL: {} -- {}".format(tool["name"], tool.get("description") or ""),
        "PARAMETERS:\n" + parameter_block(arguments),
    ]
    if folders:
        parts.append(folders_block(folders))
    parts.append("USER REQUEST:\n" + prompt.strip())
    return "\n\n".join(parts)


def advisor_system(candidates, arguments_for=None) -> str:
    return ADVISOR_SYSTEM.format(tools=candidates_block(candidates, arguments_for))


def _literal(value) -> str:
    """A value as the model should write it back: JSON, so types survive."""
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
