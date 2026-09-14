"""What every GreedyReg test needs: scans on disk, and greedy stubbed out.

No test in this suite runs a real registration. `picsl_greedy` is exercised
only as far as the child-process boundary -- the command that would be handed
to it, and the way that child is launched -- because everything this port
changed lives on this side of that line.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import sadt_greedyreg  # noqa: E402


def make_scan(path, shape=(4, 4, 4), affine=None, data=None) -> Path:
    """A small but real NIfTI on disk, readable back by nibabel."""
    import nibabel as nib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = np.zeros(shape, np.float32)
    image = nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine)
    nib.save(image, str(path))
    return path


def cohort(root, patients, t1="{p}_T1.nii.gz", t2="{p}_T2.nii.gz") -> tuple:
    """A T1 folder and a T2 folder holding one scan per patient."""
    root = Path(root)
    for patient in patients:
        make_scan(root / "t1" / t1.format(p=patient))
        make_scan(root / "t2" / t2.format(p=patient))
    return root / "t1", root / "t2"


class Recorder:
    """greedy, replaced by what it was asked and the files it would write.

    The registration invocation writes its `-o` transform; the resample writes
    the volume named third after `-rm`. Both are what the next step -- or the
    caller -- reads back, so a stub that skipped them would make the batch tests
    pass against a tool that produced nothing.
    """

    def __init__(self, fails_for=(), raises=None):
        self.commands: list = []
        self.fails_for = tuple(fails_for)
        self.raises = raises or (lambda patient: RuntimeError("greedy: convergence failed"))

    def __call__(self, command):
        command = [str(part) for part in command]
        self.commands.append(command)
        for patient in self.fails_for:
            if any(f"{patient}_" in os.path.basename(part) for part in command):
                raise self.raises(patient)
        if "-rm" in command:
            make_scan(command[command.index("-rm") + 2])
        elif "-o" in command:
            Path(command[command.index("-o") + 1]).write_text("stub transform\n")
        return ""

    @property
    def registrations(self) -> list:
        return [command for command in self.commands if "-a" in command]

    @property
    def resamples(self) -> list:
        return [command for command in self.commands if "-rm" in command]

    def registration_for(self, patient: str) -> list:
        """The affine search run for one patient, found by its `-o` transform."""
        for command in self.registrations:
            if os.path.basename(command[command.index("-o") + 1]).startswith(f"{patient}_"):
                return command
        raise AssertionError(f"no registration was run for {patient}")


@pytest.fixture
def greedy(monkeypatch) -> Recorder:
    """greedy replaced by a recorder, for the duration of one test."""
    recorder = Recorder()
    monkeypatch.setattr(sadt_greedyreg, "run_greedy", recorder)
    return recorder


@pytest.fixture
def failing_greedy(monkeypatch):
    """A recorder that raises for named patients and succeeds for the rest."""

    def build(*patients, raises=None):
        recorder = Recorder(fails_for=patients, raises=raises)
        monkeypatch.setattr(sadt_greedyreg, "run_greedy", recorder)
        return recorder

    return build
