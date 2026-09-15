"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts -- `scripts/describe.py` merges these
hints into the published schema and refuses any that name an argument or an
option the signature does not offer. Delete this file and the tool still works;
the panel just gets worse.

The one substantive decision is what is hidden. `endpoint`, `temperature` and
`timeout_seconds` describe the deployment's language model, not the clinical
request, and a panel that asks a clinician for a sampling temperature above the
box where they describe a patient's scans has its priorities backwards.
`seed` is hidden by the server's own conventions already.
"""

_REQUEST = "Request"
_CATALOGUE = "Tools"
_MODEL = "Language model"

LAYOUT = {
    "prompt": {"section": _REQUEST, "label": "What do you need?"},
    "folders": {"section": _REQUEST, "label": "Folders with your data"},
    "mode": {"section": _REQUEST, "label": "Mode"},
    "history": {"section": _REQUEST, "label": "Conversation so far", "hidden": True},

    "catalog_file": {
        "section": _CATALOGUE,
        "label": "Tool catalogue (optional)",
    },
    "candidates": {
        "section": _CATALOGUE,
        "label": "Tools shown to the model (0 = all)",
    },
    # Visible, and near the top of its own section: it is the one control that
    # decides whether anything runs on patient data. Hiding it would put an
    # approval behind a default nobody sees.
    "execute": {
        "section": _CATALOGUE,
        "label": "Run the chosen tool as well as proposing it",
        "visible_when": {"mode": "Agent (Automated)"},
    },

    "endpoint": {"section": _MODEL, "label": "Ollama endpoint", "hidden": True},
    "model_tag": {"section": _MODEL, "label": "Ollama model"},
    "temperature": {"section": _MODEL, "label": "Temperature", "hidden": True},
    # Also hidden by the SERVER's own conventions, which put `seed` in its
    # TECHNICAL set. Stated here anyway: those conventions belong to one
    # deployment, and a tool folder is meant to be servable by any of them.
    "seed": {"section": _MODEL, "label": "Sampler seed", "hidden": True},
    "timeout_seconds": {"section": _MODEL, "label": "Timeout (s)", "hidden": True},
}
