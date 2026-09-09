# FlexReg

Builds a registration patch on an intraoral surface and registers two timepoints
on it. Two arches are aligned on a REGION the clinician chooses rather than on
the whole mesh: teeth move between timepoints and the palate does not, so
registering on everything drags the result toward whatever moved most.

Ported from `FlexReg_CLI.py` and `FlexReg_Method/` in
SlicerAutomatedDentalTools.

## What moved here, and what did not

| upstream `type` | here | |
|---|---|---|
| `butterfly` | `mode="Patch"` | needs a GPU |
| `icp` | `mode="Register"`, `patch="Palate (butterfly)"` | |
| `icp_mgl` | `mode="Register"`, `patch="Mucogingival line"` | |
| `curve` | not ported | its input is a polyline drawn in the 3D view |
| `delete` | not ported | array bookkeeping, no computation |

The two that stayed behind are not oversights. `curve` takes the stroke itself,
so there is no form of it without a Slicer scene, and a mesh crossing the
network per stroke is slower than the local pass it replaces. `delete` renames
`Butterfly<n+1>` down over `Butterfly<n>`; a round trip costs more than doing it.

The GPU is not optional for the patch: upstream's propagation calls `.cuda()`
with no availability test and no device argument, which is why the module ships
191 lines of `install_pytorch.py`. Running it server-side is the point.

## Coordinates

A `.vtk` on disk is LPS and Slicer works in RAS. Upstream flipped by (-1, -1, 1)
on read and never undid it, so the file it wrote back was RAS -- invisible while
both ends are Slicer, wrong the moment anything else reads it. Here the flip is
applied on read and again on write, so a mesh leaves in the convention it
arrived in. The `.tfm` is still written in the convention SimpleITK expects.

## Verified

Patch on a real labelled arch (294,260 points): 54,030 points in the patch,
3.4s. Registering that surface onto itself returns the identity exactly --
maximum deviation 0.00e+00 on both rotation and translation.
