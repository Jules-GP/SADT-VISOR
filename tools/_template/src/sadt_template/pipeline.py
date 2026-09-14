"""The actual work, kept out of __init__.py so that run() reads as a contract.

Real tools put the ported upstream pipeline here (one module per stage if it is
large). Two things in this file are worth copying verbatim into a new tool:
`iter_scans`, which is how a batch-capable argument is expanded, and the lazy
import inside `summarise`.
"""

from pathlib import Path

from . import progress

SUFFIX = ".npy"

METRICS = {
    "mean": lambda np, values: float(np.mean(values)),
    "max": lambda np, values: float(np.max(values)),
    "min": lambda np, values: float(np.min(values)),
    "std": lambda np, values: float(np.std(values)),
}

REDUCTIONS = ("per_scan", "pooled")


class ToolInputError(ValueError):
    """An input the tool cannot work with, phrased for whoever sent it.

    The runner turns this into a 422 for the client. Anything else surfaces as
    an internal error, so raise this for bad inputs and let genuine bugs crash.
    """


def iter_scans(scans: Path) -> list[Path]:
    """Expand a batch argument into the files to process, in a stable order.

    Copied into every tool rather than shared: see CONTRIBUTING.md on why there
    is no sadt-core package. Sorting matters -- readdir order varies between
    filesystems, and an unordered batch makes two runs on the same folder
    produce differently ordered reports.
    """
    if scans.is_dir():
        found = sorted(p for p in scans.rglob(f"*{SUFFIX}") if p.is_file())
        if not found:
            raise ToolInputError(f"No {SUFFIX} file found under {scans}.")
        return found
    if scans.is_file():
        return [scans]
    raise ToolInputError(f"Input path does not exist: {scans}")


def summarise(
    scans: Path,
    output_dir: Path,
    metrics: list,
    reduction: str,
    threshold: float,
    per_scan_report: bool,
) -> Path:
    import numpy as np

    # `Literal` is published as `choices` so a client can render a picker, but
    # the runner still calls run(**params) from a JSON object -- a direct caller
    # or a stale client can send anything. Checked here, not assumed.
    unknown = [name for name in metrics if name not in METRICS]
    if unknown:
        raise ToolInputError(
            f"Unknown metric(s): {', '.join(unknown)}. Available: {', '.join(METRICS)}."
        )
    if not metrics:
        raise ToolInputError("Select at least one metric.")
    if reduction not in REDUCTIONS:
        raise ToolInputError(
            f"Unknown reduction '{reduction}'. Available: {', '.join(REDUCTIONS)}."
        )

    # Created here, not by the caller: a tool that assumes its output directory
    # already exists fails the first time it is run outside the server.
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pooled = []
    # Copied into every tool, like `iter_scans` above and for the same reason.
    # `report` is one line appended to the file the server named, and it is the
    # difference between a clinician seeing "scan 14 of 40" and seeing nothing
    # for an hour. The counter travels, the file name never does.
    found = iter_scans(scans)
    for index, scan in enumerate(found, start=1):
        progress.report(index, len(found), "scan")
        values = np.load(scan).ravel()
        kept = values[values > threshold]
        row = {"scan": scan.name, "voxels": int(kept.size)}
        # An empty selection would make np.mean warn and return nan; report the
        # count and skip the statistics instead of writing nan into a result.
        row.update(
            {name: METRICS[name](np, kept) for name in metrics} if kept.size else {}
        )
        rows.append(row)

        if reduction == "pooled":
            pooled.append(kept)

        if per_scan_report:
            (output_dir / f"{scan.stem}.txt").write_text(_format(row), encoding="utf-8")

    if reduction == "pooled":
        values = np.concatenate(pooled) if pooled else np.empty(0)
        row = {"scan": "all", "voxels": int(values.size)}
        row.update({name: METRICS[name](np, values) for name in metrics} if values.size else {})
        rows = [row]

    summary = output_dir / "summary.txt"
    summary.write_text("\n".join(_format(row) for row in rows) + "\n", encoding="utf-8")
    return summary


def _format(row: dict) -> str:
    return " ".join(f"{key}={value}" for key, value in row.items())
