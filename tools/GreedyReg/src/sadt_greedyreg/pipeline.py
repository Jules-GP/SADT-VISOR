"""Greedy affine registration of two CBCT timepoints.

Ported from `GreedyReg_CLI/GreedyReg_CLI.py`. For each patient present at both
timepoints, T2 is registered onto T1 and resampled into its frame; the
transform is written beside the resampled volume.

Greedy is called through `picsl_greedy`, which is greedy's own Python
distribution: `Greedy3D.execute(command)` takes the same argument string the
executable does, so the two invocations below are the upstream ones verbatim.
Upstream shelled out to a binary whose path was an ARGUMENT, which a client
cannot supply on a server.
"""

import logging

logger = logging.getLogger("GreedyReg")

METRICS = ("NCC", "NMI", "SSD")
TRANSFORMS = ("Rigid", "Affine")

# Degrees of freedom per transform type, as upstream chose them.
DEGREES_OF_FREEDOM = {"Rigid": "6", "Affine": "12"}


def _check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. GreedyReg optimises one of: "
            f"{', '.join(METRICS)}."
        )


def _check_transform(transform_type: str) -> None:
    if transform_type not in TRANSFORMS:
        # Named, rather than the bare `KeyError: 'rigid'` the degrees-of-freedom
        # lookup used to raise: this message is what a 422 carries back.
        raise ValueError(
            f"Unknown transform_type {transform_type!r}. GreedyReg registers "
            f"with one of: {', '.join(TRANSFORMS)}."
        )


def check_choices(metric: str, transform_type: str) -> None:
    """Refuse an unrecognised metric or transform type, naming the allowed ones.

    Called once before a batch as well as inside the command builders: a typo
    is one error before anything runs, not the same error reported forty times
    as forty failed patients.
    """
    _check_metric(metric)
    _check_transform(transform_type)


def metric_arguments(metric: str) -> list:
    """Greedy's `-m` flag. NCC carries a radius, the other two do not.

    An unrecognised metric is REFUSED rather than defaulted. The first version
    fell through to SSD, so `"ncc"` -- the same word in the wrong case, which is
    what a `sup` call or a direct API call can send -- silently changed what was
    optimised and the report said NCC anyway.
    """
    _check_metric(metric)
    if metric == "NCC":
        return ["-m", "NCC", "4x4x4"]
    if metric == "NMI":
        return ["-m", "NMI"]
    return ["-m", "SSD"]


def registration_command(fixed: str, moving: str, transform_out: str, init: str,
                         metric: str, transform_type: str, mask: str = "") -> list:
    """The affine search, verbatim from upstream's `buildRegistrationCommand`.

    `-n 100x100x50x25` is the multi-resolution schedule, `-search 100 10 20`
    the random search that precedes it. They are not exposed: they describe how
    this registration was tuned, not a per-request choice.
    """
    _check_transform(transform_type)
    command = ["-d", "3", "-a"]
    command += metric_arguments(metric)
    command += ["-i", fixed, moving]
    command += ["-o", transform_out]
    command += ["-n", "100x100x50x25"]
    command += ["-e", "0.5"]
    command += ["-search", "100", "10", "20"]
    command += ["-dof", DEGREES_OF_FREEDOM[transform_type]]
    command += ["-ia", init]
    if mask:
        command += ["-gm", mask]
    return command


def resample_command(fixed: str, moving: str, out: str, transform: str) -> list:
    return ["-d", "3", "-rf", fixed, "-rm", moving, out, "-r", transform]


def write_identity_init(path: str) -> None:
    """A 4x4 identity with the x translation nudged by 1 micron.

    Verbatim from upstream, comment included: Greedy treats an exact identity
    as "no initialisation given" and falls back to its own guess, so the nudge
    is what makes "start from where the images already are" expressible.
    """
    import numpy as np

    matrix = np.eye(4)
    matrix[0, 3] = 0.001
    with open(path, "w") as handle:
        for row in matrix:
            handle.write(" ".join(str(value) for value in row) + "\n")


def binarise_mask(source: str, destination: str) -> None:
    """Anything above zero becomes 1.0. Greedy's `-gm` wants a float mask."""
    import nibabel as nib
    import numpy as np

    mask = nib.load(source)
    data = (mask.get_fdata() > 0).astype(np.float32)
    written = nib.Nifti1Image(data, mask.affine)
    written.header.set_data_dtype(np.float32)
    nib.save(written, destination)


# How long one registration may run before it is abandoned. Upstream bounded
# each greedy call with `subprocess.run(..., timeout=600)`; dropping to an
# in-process `Greedy3D.execute()` would have removed that bound silently, and
# the server's own TOOL_TIMEOUT_SECONDS defaults to 0 -- "none", because a
# cohort legitimately takes hours. So one pathological pair could hold a
# concurrency slot for ever. Greedy is therefore still run as a SUBPROCESS,
# with its own interpreter: the package replaces the binary, not the boundary.
CASE_TIMEOUT_SECONDS = 600

_CHILD = """
import sys
from picsl_greedy import Greedy3D
Greedy3D().execute(" ".join(sys.argv[1:]), out=sys.stdout)
"""


def run_greedy(command: list, timeout: float = CASE_TIMEOUT_SECONDS) -> str:
    """One greedy invocation, in a child process. Returns whatever it printed.

    A non-zero exit carries greedy's own message, which is the one a caller
    needs: nothing this tool knows explains why a registration did not
    converge. A timeout is raised as such rather than as a generic failure,
    because the two call for different things -- a bigger bound, or different
    images.
    """
    import subprocess
    import sys

    finished = subprocess.run(
        [sys.executable, "-c", _CHILD, *[str(part) for part in command]],
        capture_output=True, text=True, timeout=timeout,
    )
    if finished.returncode != 0:
        raise RuntimeError(
            finished.stderr.strip() or finished.stdout.strip() or "greedy failed"
        )
    return finished.stdout
