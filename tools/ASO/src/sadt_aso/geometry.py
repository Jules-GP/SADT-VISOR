"""The landmark-registration maths shared by both ASO engines.

`ASO_CBCT_utils/utils.py` and `ASO_IOS_utils/{icp,transformation}.py` carried
two copies of all of this -- `RotationMatrix`, `AngleAndAxisVectors`,
`InitICP`, `FindOptimalLandmarks`, `ComputeMeanDistance` -- that had drifted
apart in ways nobody could have wanted (the IOS copy cached its input to a
`.npy` file on disk and reloaded it on every search iteration; the CBCT copy
did not). One implementation, parameterised where the two genuinely differ.

Pure numpy: no VTK, no SimpleITK, no I/O. That is what makes it testable
without either heavy library, and what keeps the triplet search free of the
global state it used to depend on.

**Nothing in this module rounds, reassociates or approximates.** The triplet
search evaluates every ordered triplet, and the arithmetic it evaluates them
with has to be the arithmetic `init_icp` will redo on the winner -- otherwise
the search and the transform disagree about which triplet is best, silently,
in the last bits. So the speedups here are of exactly two kinds:

* **the same expression, computed once instead of n times.** The coarse
  alignment's first half depends on the first two landmarks alone, so the
  search builds it once per PAIR (`_Alignment`) and finishes it once per
  triplet; the target-side vectors depend on the reference alone
  (`_TargetPairs`). Caching a deterministic function of unchanged inputs
  cannot move a bit.
* **numpy's own fast path, called directly.** `_norm` is the two operations
  `np.linalg.norm` performs for a real 1-D vector and `_cross3` is the six
  `np.cross` performs for two 3-vectors -- written out, not reimplemented, so
  the values are identical and only the dispatch is gone.

`tests/test_geometry_equivalence.py` holds both halves to that claim against
slow reference implementations, on random inputs, asserting bit-identity.
"""

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Steps init_icp reports, so a caller that needs SimpleITK transforms (the CBCT
# engine composes them to resample the volume) can rebuild them in the same
# order without this module importing SimpleITK.
STEP_TRANSLATION = "translation"
STEP_ROTATION = "rotation"

# Copied rather than rebuilt. `np.eye` spends most of its time in Python, and
# the triplet search asks for one identity per triplet; a copy of a constant is
# the same array for a fifth of the cost. Never handed out uncopied.
_EYE3 = np.eye(3)
_EYE4 = np.eye(4)

# Distinguishes "not cached yet" from a cached None (`_scaled` returns None for
# a vector it cannot scale, and that answer is worth caching too).
_MISSING = object()


def _eye3() -> np.ndarray:
    return _EYE3.copy()


def _eye4() -> np.ndarray:
    return _EYE4.copy()


def _norm(vector) -> np.float64:
    """`np.linalg.norm(vector)` for a real vector, without the dispatch.

    numpy's own fast path for `ord=None` is `sqrt(x.dot(x))` after casting an
    integer array to float (numpy/linalg/_linalg.py), and `ravel(order='K')` is
    a no-op on the 1-D vectors this module has. So this is not an
    approximation of `norm`, it is the two floating-point operations `norm`
    would have performed on the same values -- which matters, because the
    triplet search calls it once per landmark per triplet and `norm` spends
    ten times longer deciding which norm was asked for than computing it.
    """
    vector = np.asarray(vector)
    if not issubclass(vector.dtype.type, np.inexact):
        vector = vector.astype(float)
    return np.sqrt(vector.dot(vector))


def _cross3(first, second) -> np.ndarray:
    """`np.cross(first, second)` for two 3-vectors, written out.

    numpy computes exactly these three differences of two products
    (numpy/_core/numeric.py); it reaches them through `moveaxis`,
    `normalize_axis_index` and a broadcast shape negotiation that cost more
    than the arithmetic by an order of magnitude.
    """
    return np.array(
        [
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        ]
    )


def rotation_matrix(axis, theta: float) -> np.ndarray:
    """Rotation matrix for a counterclockwise rotation of `theta` radians
    about `axis` (Euler-Rodrigues).

    A zero-length axis means the two vectors were already parallel, so there is
    nothing to rotate: the original divided by `np.linalg.norm(axis)` anyway and
    produced a matrix of NaN, which then propagated silently into the final
    transform and came out as a scan full of zeros.
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = _norm(axis)
    if norm == 0 or not np.isfinite(norm):
        return _eye3()
    axis = axis / norm
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    matrix = np.empty((3, 3))
    matrix[0, 0] = aa + bb - cc - dd
    matrix[0, 1] = 2 * (bc + ad)
    matrix[0, 2] = 2 * (bd - ac)
    matrix[1, 0] = 2 * (bc - ad)
    matrix[1, 1] = aa + cc - bb - dd
    matrix[1, 2] = 2 * (cd + ab)
    matrix[2, 0] = 2 * (bd + ac)
    matrix[2, 1] = 2 * (cd - ab)
    matrix[2, 2] = aa + dd - bb - cc
    return matrix


def _scaled(vector):
    """`(vector / max(vector), that vector's length)`, or None.

    The two ways this comes out with nothing to say, both of which the original
    answered with NaN:

    * `np.amax(v)` is used to scale the vector (it is what the original did,
      and the scaling cancels out of the angle) -- but it is zero for a null
      vector and negative for an all-negative one. Callers pass componentwise
      absolute values, so a zero only happens for coincident landmarks.
    * a scaled vector of length zero, which is the same case reached one step
      later.
    """
    vector = np.asarray(vector, dtype=np.float64)
    scale = np.amax(vector)
    if scale == 0:
        return None
    unit = vector / scale
    norm = _norm(unit)
    if norm == 0:
        return None
    return unit, norm


def _angle_and_axis(first, second) -> tuple:
    """Angle and rotation axis between two already-`_scaled` vectors.

    `np.arccos` of a dot product that floating-point rounding pushed to
    1.0000000002 is NaN. `pre_icp.py` clamped the upper end only, leaving
    antiparallel vectors to produce NaN.
    """
    if first is None or second is None:
        return 0.0, np.zeros(3)
    first_unit, first_norm = first
    second_unit, second_norm = second
    angle = np.arccos(
        np.clip(first_unit.dot(second_unit) / (first_norm * second_norm), -1.0, 1.0)
    )
    return angle, _cross3(first_unit, second_unit)


def angle_and_axis(v1, v2) -> tuple:
    """Angle and rotation axis taking `v1` onto `v2`."""
    return _angle_and_axis(_scaled(v1), _scaled(v2))


def translate_landmarks(landmarks: dict, offset) -> dict:
    return {name: point + offset for name, point in landmarks.items()}


def transform_landmarks(landmarks: dict, matrix: np.ndarray) -> dict:
    """Apply a 4x4 homogeneous matrix to every point of a landmark dict.

    The homogeneous buffer is filled in place rather than rebuilt per point:
    `np.append` allocates twice and goes through `concatenate` and `ravel` to
    put a 1.0 on the end of a three-vector, which cost more than the
    matrix-vector product it was preparing. The four numbers handed to the
    product are the same four numbers.
    """
    homogeneous = np.empty(4)
    homogeneous[3] = 1.0
    transformed = {}
    for name, point in landmarks.items():
        homogeneous[:3] = point
        transformed[name] = (matrix @ homogeneous)[:3]
    return transformed


def _mean_distance_over(shared: list, source: dict, target: dict) -> float:
    if not shared:
        return float("inf")
    return float(np.mean([_norm(source[name] - target[name]) for name in shared]))


def mean_distance(source: dict, target: dict) -> float:
    """Mean point-to-point distance over the labels the two dicts share."""
    return _mean_distance_over(
        [name for name in source if name in target], source, target
    )


class _TargetPairs:
    """`_scaled(|target[other] - target[first]|)`, once per ordered pair.

    The coarse alignment asks for this twice -- (first, second) and
    (first, third) -- and the search walks every ordered triplet, so each pair
    is asked for up to n-2 times for a target that does not move. Same
    subtraction, same absolute value, same scaling; computed once.
    """

    __slots__ = ("_target", "_cache")

    def __init__(self, target: dict):
        self._target = target
        self._cache: dict = {}

    def __call__(self, first: str, other: str):
        key = (first, other)
        value = self._cache.get(key, _MISSING)
        if value is _MISSING:
            value = _scaled(np.absolute(self._target[other] - self._target[first]))
            self._cache[key] = value
        return value


class _Alignment:
    """The coarse alignment of a source landmark set onto a target one, split
    where it stops depending on the third landmark.

    Building it performs the translation onto `first` and the rotation aligning
    the first->second direction; `finish` adds the rotation aligning
    first->third about the first->second axis, which is what makes the
    alignment specific to a triplet.

    The split exists because the triplet search evaluates n*(n-1)*(n-2)
    triplets and the first half of each one depends on n*(n-1) pairs. Nothing
    is approximated to make it: the values `finish` works from are the values
    the unsplit function had at the same point.
    """

    __slots__ = ("target", "first", "second", "offset", "rotation", "aligned", "axis", "_pairs")

    def __init__(self, source: dict, target: dict, first: str, second: str, pairs=None):
        self.target = target
        self.first = first
        self.second = second
        self._pairs = pairs if pairs is not None else _TargetPairs(target)

        # ----- translation onto the first pick
        self.offset = target[first] - source[first]
        moved = translate_landmarks(source, self.offset)

        # ----- rotation aligning the first->second direction
        angle, axis = _angle_and_axis(
            self._pairs(first, second),
            _scaled(np.absolute(moved[second] - moved[first])),
        )
        self.rotation = _eye4()
        self.rotation[:3, :3] = rotation_matrix(axis, angle)
        self.aligned = transform_landmarks(moved, self.rotation)

        # The axis the second rotation turns about: the first->second direction
        # as it now stands, which the third landmark does not change.
        self.axis = np.absolute(self.aligned[second] - self.aligned[first])

    def finish(self, third: str) -> tuple:
        """`(transformed source, the second rotation as a 4x4)` for `third`.

        The rotation is about the first->second axis so the alignment already
        obtained is not undone.
        """
        angle, _ = _angle_and_axis(
            self._pairs(self.first, third),
            _scaled(np.absolute(self.aligned[third] - self.aligned[self.first])),
        )
        rotation = _eye4()
        rotation[:3, :3] = rotation_matrix(self.axis, angle)
        return transform_landmarks(self.aligned, rotation), rotation


def init_icp(source: dict, target: dict, picks, include_translation: bool) -> tuple:
    """Coarse alignment from three landmarks: one translation then two rotations.

    Returns `(transformed_source, matrix, steps)`.

    `include_translation` is the one real difference between the two engines,
    and it is deliberate on both sides. The CBCT engine leaves it out: both
    volumes have already been recentred on the physical origin, so re-adding a
    landmark-derived offset would push the scan back off centre -- which is
    also why it drops the ICP translation afterwards. The IOS engine keeps it:
    two intra-oral scans share no common origin.
    """
    first, second, third = picks
    alignment = _Alignment(source, target, first, second)
    transformed, second_rotation = alignment.finish(third)

    steps = [
        (STEP_TRANSLATION, alignment.offset),
        (STEP_ROTATION, alignment.rotation[:3, :3]),
        (STEP_ROTATION, second_rotation[:3, :3]),
    ]

    matrix = _eye4()
    if include_translation:
        matrix[:3, 3] = alignment.offset
    matrix = alignment.rotation @ matrix
    matrix = second_rotation @ matrix

    return transformed, matrix, steps


def best_triplet(
    source: dict,
    target: dict,
    include_translation: bool,
    max_triplets: int,
    seed: int,
) -> tuple:
    """The three landmarks whose coarse alignment leaves the smallest mean
    distance, searched deterministically.

    The original drew triplets with the GLOBAL `np.random`, so the same request
    gave a different orientation each time it ran, and two concurrent requests
    consumed each other's random state. An orientation applied to patient data
    has to be reproducible, so:

    * every ordered triplet is evaluated when there are at most `max_triplets`
      of them -- which covers any realistic selection (7 landmarks is 210,
      14 is 2184) and is both faster and better than sampling;
    * above that the candidates are sampled from a LOCAL generator seeded from
      configuration, so a rerun repeats the same search.

    `include_translation` is accepted and not used. What ranks a triplet is the
    mean distance its coarse alignment leaves, and `init_icp` moves the points
    identically either way -- the flag decides only whether the translation is
    folded into the matrix it returns, which the search discards.
    `test_the_triplet_search_ignores_the_translation_flag` pins that.
    """
    labels = list(source)
    total = len(labels) * (len(labels) - 1) * (len(labels) - 2)

    if total <= max_triplets:
        candidates = itertools.permutations(labels, 3)
    else:
        rng = np.random.default_rng(seed)
        candidates = _sampled_triplets(labels, max_triplets, rng)

    # The labels scored, hoisted out of the loop: every candidate transforms
    # the same source keys, so `mean_distance` recomputed this list once per
    # triplet and got the same answer every time.
    shared = [name for name in source if name in target]
    pairs = _TargetPairs(target)
    alignments: dict = {}

    best, best_distance = None, float("inf")
    for picks in candidates:
        first, second, third = picks
        alignment = alignments.get((first, second))
        if alignment is None:
            alignment = _Alignment(source, target, first, second, pairs)
            alignments[(first, second)] = alignment
        transformed, _ = alignment.finish(third)
        distance = _mean_distance_over(shared, transformed, target)
        if distance < best_distance:
            best, best_distance = picks, distance

    if best is None:  # fewer than three landmarks; callers check first
        raise ValueError("Need at least three landmarks to search for a triplet")
    logger.debug("Best triplet leaves a mean distance of %.3f", best_distance)
    return best


def _sampled_triplets(labels: list, count: int, rng) -> list:
    seen, picks = set(), []
    while len(picks) < count:
        candidate = tuple(rng.choice(labels, size=3, replace=False))
        if candidate not in seen:
            seen.add(candidate)
            picks.append(candidate)
    return picks
