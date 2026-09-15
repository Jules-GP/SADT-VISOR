"""Talking to an Ollama endpoint, over its HTTP API, with the standard library.

Three things this deliberately does not do, each of which upstream did:

- **It does not install Ollama.** `Agent/Agent.py:101-207` downloads a
  500 MB-1 GB binary over plain `urllib` with no checksum and no signature,
  `chmod 0o755`s it, strips the macOS Gatekeeper quarantine attribute, and
  `Popen`s it -- and on Linux it `tarfile.extractall`s a downloaded archive with
  no member sanitisation at all. A packaged tool declares a reachable endpoint
  as a prerequisite; it does not acquire a server.
- **It does not pull the model.** `utils.chat_with_auto_pull` treated a 404 as a
  cue to download ~5 GB from inside a request handler. A server holding
  confidential imaging does not make outbound calls mid-request, and a request
  that blocks for a quarter of an hour on a first run is not a request.
- **It does not use the `ollama` client package.** The two calls this tool
  makes are one POST each; going through the standard library keeps this
  tool's dependency list empty, which is the whole point of the ranker
  measurement in README.md.

`format="json"` asks Ollama to constrain the answer to a JSON object, which the
router and the extractor both need. Reasoning models still wrap it, so
`parse_json_object` unwraps rather than trusting.
"""

import json
import os
import re
import urllib.error
import urllib.request

from .errors import ToolInputError, ToolUnavailableError

# Where Ollama listens when nothing says otherwise. `OLLAMA_HOST` is Ollama's
# own convention and a property of the DEPLOYMENT, like `SADT_API` -- not a
# setting that should have become an argument, because it says where a service
# is rather than what the tool should do. The `endpoint` argument overrides it
# per request either way.
ENDPOINT_ENV = "OLLAMA_HOST"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"

# The model stays resident between the router call and the extraction call.
# Reloading 8B of weights costs several seconds and this tool makes two calls in
# a row, so the default 5 minutes would already be enough -- 30m covers a
# clinician typing a follow-up message.
KEEP_ALIVE = "30m"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def resolve_endpoint(endpoint: str) -> str:
    """The endpoint to talk to: the argument, then `OLLAMA_HOST`, then loopback."""
    chosen = (endpoint or "").strip() or os.environ.get(ENDPOINT_ENV, "").strip()
    chosen = chosen or DEFAULT_ENDPOINT
    if "://" not in chosen:
        # `OLLAMA_HOST=127.0.0.1:11434` is how Ollama's own documentation
        # spells it, and urllib will not open that.
        chosen = "http://" + chosen
    return chosen.rstrip("/")


def chat(endpoint: str, model: str, messages, *, json_format: bool = False,
         temperature: float = 0.0, seed: int = 0, timeout_seconds: int = 300) -> str:
    """One `/api/chat` round trip. Returns the assistant's message content."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": float(temperature), "seed": int(seed)},
    }
    if json_format:
        body["format"] = "json"

    request = urllib.request.Request(
        endpoint + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise _http_failure(exc, endpoint, model)
    except (urllib.error.URLError, OSError) as exc:
        raise ToolUnavailableError(
            "No Ollama server answered at '{}' ({}). This tool needs a running "
            "endpoint; it does not install or start one. Start Ollama, or pass "
            "`endpoint`.".format(endpoint, exc)
        )
    except ValueError as exc:
        raise ToolUnavailableError(
            "'{}' did not answer with JSON: {}.".format(endpoint, exc)
        )

    message = (payload or {}).get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ToolUnavailableError(
            "'{}' answered without a message content field.".format(endpoint)
        )
    return content


def _http_failure(exc, endpoint: str, model: str):
    """Turn an HTTP error into the message whose fix is the actual fix."""
    try:
        detail = exc.read().decode("utf-8", "replace")[:500]
    except Exception:  # noqa: BLE001 - the status is the useful half either way
        detail = ""
    if exc.code == 404 or "not found" in detail.lower():
        # The model tag is quoted WHOLE. Upstream's remediation hint did
        # `model_name.split(':')[0]` (`Agent/Agent.py:915`), so a missing
        # `qwen3:8b` was reported as "run: ollama pull qwen3" -- a different
        # model, several gigabytes, and still not the one the tool asks for.
        return ToolUnavailableError(
            "The model '{}' is not available on the Ollama server at '{}'. Pull "
            "it on that machine with: ollama pull {}. This tool does not pull "
            "models: a ~5 GB download inside a request is not a request.".format(
                model, endpoint, model
            )
        )
    return ToolUnavailableError(
        "The Ollama server at '{}' answered {} {}. {}".format(
            endpoint, exc.code, exc.reason, detail
        ).strip()
    )


def parse_json_object(text: str, what: str) -> dict:
    """The one JSON object in a model's answer, or a message saying there is none.

    `format="json"` makes Ollama constrain the answer, but a reasoning model
    still emits a `<think>` block around it and some wrap it in a fence, so both
    are removed before the first balanced object is taken.
    """
    cleaned = _FENCE.sub("", _THINK.sub("", text or "")).strip()
    try:
        parsed = json.loads(cleaned)
    except ValueError:
        parsed = _first_object(cleaned)
    if not isinstance(parsed, dict):
        raise ToolInputError(
            "The model's {} answer was not a JSON object. It said: {!r}".format(
                what, (text or "")[:300]
            )
        )
    return parsed


def _first_object(text: str):
    """The first balanced `{...}` in `text`, parsed, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None
