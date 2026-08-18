# sadt-areg-common

What `AREG_CBCT` and `AREG_IOS` must not disagree about.

## Why these four and not the others

The split duplicated both engines' implementations and shared almost nothing —
that is the standing rule, and it is right. These four are the exception, and
each earns it differently.

**`pairing.py` is the one that matters.** It was nearly duplicated on the
reasoning that "the two modes pair different things — volumes against meshes",
which sounds obviously true and is wrong. The functions that really are
modality-specific, `pair()` and `discover()`, are called by **neither engine**:
they belong to the dispatcher, which the split separated anyway. What the
engines actually share is `patient_stem()` — how a patient's identity is derived
from a filename, by stripping timepoint and jaw tokens.

That is a **convention**, and a divergence in it does not fail: it makes
`AREG_CBCT` and `AREG_IOS` derive different keys from the same name, so a
cross-modality registration pairs a patient with almost-themselves and says
nothing. `AREG_IOSCBCT` will lean on it from both sides at once, which is what
settles it. `is_previous_output()` is the same family: two copies that drift
means one tool re-ingesting what the other produced.

**`catalogs.py`** holds the modality and automation tables. Published in the
schema, keyed on by the server, and read by both — a second copy is a panel
offering a mode the tool no longer has.

**`scans.py`** is the file-extension vocabulary, the same contract with the
outside world that `ALI/common/discovery.py` carries.

**`errors.py`** is here rather than duplicated only because it is three lines
and travels with `pairing`'s raises. Note that ALI keeps its copy duplicated:
errors cross the process boundary by class NAME, so sharing the class is never
the mechanism — it is a convenience here, not a requirement.

## The constraint

`dependencies = []`, and it has to stay that way. This installs into two
environments whose pins are deliberately incompatible; anything it pulled in
would have to satisfy both at once, which is exactly what the split removed.
