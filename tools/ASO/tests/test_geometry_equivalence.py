"""The registration maths, held bit-for-bit to the implementation it replaced.

`geometry.py` was made faster and nothing else. What licenses that claim is not
that the fast version looks like the slow one, it is that on random inputs the
two produce the SAME FLOATING-POINT NUMBERS -- not close ones. A coarse
alignment that differs in the last bit picks a different triplet on a
near-tie, and a different triplet is a different orientation of a patient's
skull.

So this file carries the reference implementation: `geometry.py` exactly as it
stood before, transcribed, using `np.linalg.norm`, `np.cross`, `np.append` and
`np.eye` where the fast version uses its own. Every test here asserts
`np.array_equal` or `==`, never `allclose`.

The two places the rewrite could have moved a value, and what pins them:

* `_norm` claims to be what `np.linalg.norm` does for a real vector, and
  `_cross3` what `np.cross` does for two 3-vectors --
  `test_the_norm_shortcut_is_numpys_own_answer` and
  `test_the_cross_shortcut_is_numpys_own_answer`.
* the triplet search now computes the first half of each alignment once per
  PAIR instead of once per triplet, and the target-side vectors once per pair
  instead of once per triplet -- `test_init_icp_matches_the_reference_exactly`
  and `test_the_triplet_search_matches_the_reference_exactly`.
"""

import itertools

import numpy as np
import pytest

from sadt_aso import geometry


# ---------------------------------------------------------------------------
# The reference implementation: geometry.py as it stood before the rewrite.
# ---------------------------------------------------------------------------

def reference_rotation_matrix(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0 or not np.isfinite(norm):
        return np.eye(3)
    axis = axis / norm
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def reference_angle_and_axis(v1, v2):
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    scale1, scale2 = np.amax(v1), np.amax(v2)
    if scale1 == 0 or scale2 == 0:
        return 0.0, np.zeros(3)
    v1_u = v1 / scale1
    v2_u = v2 / scale2
    norm1, norm2 = np.linalg.norm(v1_u), np.linalg.norm(v2_u)
    if norm1 == 0 or norm2 == 0:
        return 0.0, np.zeros(3)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u) / (norm1 * norm2), -1.0, 1.0))
    return angle, np.cross(v1_u, v2_u)


def reference_transform_landmarks(landmarks, matrix):
    return {
        name: (matrix @ np.append(point, 1.0))[:3] for name, point in landmarks.items()
    }


def reference_mean_distance(source, target):
    shared = [name for name in source if name in target]
    if not shared:
        return float("inf")
    return float(
        np.mean([np.linalg.norm(source[name] - target[name]) for name in shared])
    )


def reference_init_icp(source, target, picks, include_translation):
    first, second, third = picks
    steps = []
    matrix = np.eye(4)

    offset = target[first] - source[first]
    translation = np.eye(4)
    translation[:3, 3] = offset
    steps.append((geometry.STEP_TRANSLATION, offset))
    source = {name: point + offset for name, point in source.items()}
    if include_translation:
        matrix = translation

    angle, axis = reference_angle_and_axis(
        np.absolute(target[second] - target[first]),
        np.absolute(source[second] - source[first]),
    )
    rotation = np.eye(4)
    rotation[:3, :3] = reference_rotation_matrix(axis, angle)
    steps.append((geometry.STEP_ROTATION, rotation[:3, :3]))
    source = reference_transform_landmarks(source, rotation)
    matrix = rotation @ matrix

    angle, axis = reference_angle_and_axis(
        np.absolute(target[third] - target[first]),
        np.absolute(source[third] - source[first]),
    )
    rotation = np.eye(4)
    rotation[:3, :3] = reference_rotation_matrix(
        np.absolute(source[second] - source[first]), angle
    )
    steps.append((geometry.STEP_ROTATION, rotation[:3, :3]))
    source = reference_transform_landmarks(source, rotation)
    matrix = rotation @ matrix

    return source, matrix, steps


def reference_best_triplet(source, target, include_translation, max_triplets, seed):
    labels = list(source)
    total = len(labels) * (len(labels) - 1) * (len(labels) - 2)
    if total <= max_triplets:
        candidates = itertools.permutations(labels, 3)
    else:
        rng = np.random.default_rng(seed)
        candidates = geometry._sampled_triplets(labels, max_triplets, rng)

    best, best_distance = None, float("inf")
    for picks in candidates:
        transformed, _, _ = reference_init_icp(
            source, target, picks, include_translation
        )
        distance = reference_mean_distance(transformed, target)
        if distance < best_distance:
            best, best_distance = picks, distance
    if best is None:
        raise ValueError("Need at least three landmarks to search for a triplet")
    return best


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_pair(rng, count, scale=40.0, rotate=True):
    """A target landmark set and a source one that is a rigid move of it."""
    target = {f"L{index}": rng.normal(scale=scale, size=3) for index in range(count)}
    if not rotate:
        return dict(target), target
    matrix = geometry.rotation_matrix(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
    offset = rng.normal(scale=10.0, size=3)
    source = {name: matrix @ point + offset for name, point in target.items()}
    return source, target


def _identical_landmarks(first, second):
    if set(first) != set(second):
        return False
    return all(np.array_equal(first[name], second[name]) for name in first)


# ---------------------------------------------------------------------------
# the two shortcuts into numpy
# ---------------------------------------------------------------------------

def test_the_norm_shortcut_is_numpys_own_answer():
    """`_norm` is `np.linalg.norm`'s fast path called directly, so it has to
    return the identical double -- not one within a tolerance."""
    rng = np.random.default_rng(7)
    for _ in range(2000):
        vector = rng.normal(scale=10.0 ** rng.integers(-6, 7), size=3)
        assert _bits(geometry._norm(vector)) == _bits(np.linalg.norm(vector))
    for vector in (
        np.zeros(3),
        np.array([1e-300, 0.0, 0.0]),
        np.array([1e150, 1e150, 1e150]),
        np.array([-3.0, 4.0, 12.0]),
    ):
        assert _bits(geometry._norm(vector)) == _bits(np.linalg.norm(vector))


def test_the_norm_shortcut_casts_integers_the_way_numpy_does():
    """`np.linalg.norm` casts an integer array to float before squaring it, and
    an integer dot product would overflow where a float one saturates."""
    vector = np.array([3, 4, 12])
    assert _bits(geometry._norm(vector)) == _bits(np.linalg.norm(vector))


def test_the_cross_shortcut_is_numpys_own_answer():
    """`_cross3` is the three differences of products `np.cross` computes."""
    rng = np.random.default_rng(11)
    for _ in range(2000):
        first = rng.normal(scale=10.0 ** rng.integers(-4, 5), size=3)
        second = rng.normal(scale=10.0 ** rng.integers(-4, 5), size=3)
        assert np.array_equal(geometry._cross3(first, second), np.cross(first, second))


def _bits(value) -> int:
    """The exact bit pattern of a double, so two NaNs compare equal and 0.0 and
    -0.0 do not."""
    return int(np.float64(value).view(np.int64))


# ---------------------------------------------------------------------------
# the primitives
# ---------------------------------------------------------------------------

def test_rotation_matrix_matches_the_reference_exactly():
    """The matrix is now filled element by element instead of being built from
    a nested list. Same expressions, same order, same doubles."""
    rng = np.random.default_rng(3)
    for _ in range(3000):
        axis = rng.normal(size=3)
        theta = rng.uniform(-2 * np.pi, 2 * np.pi)
        assert np.array_equal(
            geometry.rotation_matrix(axis, theta),
            reference_rotation_matrix(axis, theta),
        )


@pytest.mark.parametrize(
    "axis", [np.zeros(3), np.array([np.nan, 0.0, 0.0]), np.array([np.inf, 1.0, 0.0])]
)
def test_rotation_matrix_matches_the_reference_on_degenerate_axes(axis):
    """The guard that stops a NaN matrix reaching the resampler has to fire on
    the same inputs it fired on before."""
    assert np.array_equal(
        geometry.rotation_matrix(axis, 0.5), reference_rotation_matrix(axis, 0.5)
    )


def test_angle_and_axis_matches_the_reference_exactly():
    rng = np.random.default_rng(5)
    for _ in range(3000):
        first = np.absolute(rng.normal(scale=30.0, size=3))
        second = np.absolute(rng.normal(scale=30.0, size=3))
        angle, axis = geometry.angle_and_axis(first, second)
        expected_angle, expected_axis = reference_angle_and_axis(first, second)
        assert _bits(angle) == _bits(expected_angle)
        assert np.array_equal(axis, expected_axis)


@pytest.mark.parametrize(
    "first,second",
    [
        (np.zeros(3), np.array([1.0, 2.0, 3.0])),
        (np.array([1.0, 2.0, 3.0]), np.zeros(3)),
        (np.array([-1.0, -2.0, -3.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])),
        (np.array([1, 2, 3]), np.array([3, 2, 1])),
    ],
)
def test_angle_and_axis_matches_the_reference_on_the_degenerate_cases(first, second):
    """A null vector, an all-negative one, two parallel ones and an integer
    pair: the four ways the scaling could stop meaning anything."""
    angle, axis = geometry.angle_and_axis(first, second)
    expected_angle, expected_axis = reference_angle_and_axis(first, second)
    assert _bits(angle) == _bits(expected_angle)
    assert np.array_equal(axis, expected_axis)


def test_transform_landmarks_matches_the_reference_exactly():
    """The homogeneous vector is filled in place instead of built by
    `np.append`; the four numbers reaching the product are the same four."""
    rng = np.random.default_rng(13)
    for _ in range(500):
        landmarks = {f"L{index}": rng.normal(scale=50.0, size=3) for index in range(9)}
        matrix = np.eye(4)
        matrix[:3, :3] = geometry.rotation_matrix(rng.normal(size=3), rng.uniform(0, 3))
        matrix[:3, 3] = rng.normal(scale=20.0, size=3)
        assert _identical_landmarks(
            geometry.transform_landmarks(landmarks, matrix),
            reference_transform_landmarks(landmarks, matrix),
        )


def test_transform_landmarks_does_not_alias_its_results():
    """The reused buffer must not leak into the results: every landmark keeps
    its own coordinates, which a shared view would not."""
    landmarks = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([4.0, 5.0, 6.0])}
    moved = geometry.transform_landmarks(landmarks, np.eye(4))
    assert np.array_equal(moved["A"], [1.0, 2.0, 3.0])
    assert np.array_equal(moved["B"], [4.0, 5.0, 6.0])
    moved["A"][0] = 99.0
    assert np.array_equal(moved["B"], [4.0, 5.0, 6.0])
    assert np.array_equal(landmarks["A"], [1.0, 2.0, 3.0])


def test_mean_distance_matches_the_reference_exactly():
    rng = np.random.default_rng(17)
    for _ in range(500):
        source, target = _random_pair(rng, 8)
        assert _bits(geometry.mean_distance(source, target)) == _bits(
            reference_mean_distance(source, target)
        )


def test_mean_distance_still_reports_infinity_for_disjoint_sets():
    assert geometry.mean_distance({"A": np.zeros(3)}, {"B": np.zeros(3)}) == float("inf")


# ---------------------------------------------------------------------------
# the coarse alignment and the search
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("include_translation", [False, True])
def test_init_icp_matches_the_reference_exactly(include_translation):
    """Splitting the alignment at the third landmark must not move it: the
    transformed points, the 4x4 and every reported step, bit for bit."""
    rng = np.random.default_rng(19)
    for _ in range(300):
        source, target = _random_pair(rng, 7)
        picks = tuple(rng.permutation(list(source))[:3])
        moved, matrix, steps = geometry.init_icp(
            source, target, picks, include_translation
        )
        expected_moved, expected_matrix, expected_steps = reference_init_icp(
            source, target, picks, include_translation
        )
        assert _identical_landmarks(moved, expected_moved)
        assert np.array_equal(matrix, expected_matrix)
        assert [kind for kind, _ in steps] == [kind for kind, _ in expected_steps]
        for (_, payload), (_, expected) in zip(steps, expected_steps):
            assert np.array_equal(payload, expected)


def test_init_icp_matches_the_reference_on_coincident_landmarks():
    """Two landmarks at the same point make the scaling degenerate, which is
    the branch that used to produce NaN. Both versions must take it."""
    target = {
        "A": np.array([0.0, 0.0, 0.0]),
        "B": np.array([0.0, 0.0, 0.0]),
        "C": np.array([1.0, 2.0, 3.0]),
        "D": np.array([-4.0, 5.0, 1.0]),
    }
    source = {name: point + np.array([7.0, -2.0, 3.0]) for name, point in target.items()}
    for picks in itertools.permutations(list(source), 3):
        moved, matrix, _ = geometry.init_icp(source, target, picks, False)
        expected_moved, expected_matrix, _ = reference_init_icp(
            source, target, picks, False
        )
        assert _identical_landmarks(moved, expected_moved)
        assert np.array_equal(matrix, expected_matrix)


@pytest.mark.parametrize("count", [3, 4, 6, 7, 9])
def test_the_triplet_search_matches_the_reference_exactly(count):
    """The search now evaluates each PAIR's half-alignment once instead of once
    per triplet. It has to pick the same triplet -- including the tie-breaking,
    which is 'the first candidate in permutation order strictly beating the
    best so far'."""
    rng = np.random.default_rng(23 + count)
    for _ in range(40):
        source, target = _random_pair(rng, count)
        assert geometry.best_triplet(source, target, False, 2500, 0) == \
            reference_best_triplet(source, target, False, 2500, 0)


def test_the_triplet_search_matches_the_reference_when_it_samples():
    """Above `max_triplets` the candidates come from a seeded generator; the
    two versions must walk the same sample in the same order."""
    rng = np.random.default_rng(97)
    source, target = _random_pair(rng, 14)
    for max_triplets in (50, 200):
        assert geometry.best_triplet(source, target, True, max_triplets, 5) == \
            reference_best_triplet(source, target, True, max_triplets, 5)


def test_the_triplet_search_matches_the_reference_on_a_perfect_fit():
    """Source and target identical: every triplet leaves distance 0, so every
    comparison is a tie and only the iteration order decides. This is the case
    a reordered search would silently get wrong."""
    rng = np.random.default_rng(29)
    source, target = _random_pair(rng, 6, rotate=False)
    assert geometry.best_triplet(source, target, False, 2500, 0) == \
        reference_best_triplet(source, target, False, 2500, 0)


def test_the_triplet_search_ignores_the_translation_flag():
    """`include_translation` decides only whether the translation is folded
    into the matrix `init_icp` returns, and the search scores the transformed
    POINTS, which move identically either way. The flag is kept in the
    signature because both engines pass it; this is what says it cannot change
    the answer."""
    rng = np.random.default_rng(31)
    for _ in range(20):
        source, target = _random_pair(rng, 7)
        assert geometry.best_triplet(source, target, False, 2500, 0) == \
            geometry.best_triplet(source, target, True, 2500, 0)


def test_the_search_and_the_final_alignment_agree_on_the_winner():
    """The search's score for the triplet it returns has to be the score
    `init_icp` reproduces on that triplet -- the search is only meaningful
    because the two are the same computation."""
    rng = np.random.default_rng(37)
    for _ in range(50):
        source, target = _random_pair(rng, 7)
        picks = geometry.best_triplet(source, target, False, 2500, 0)
        moved, _, _ = geometry.init_icp(source, target, picks, False)
        best = geometry.mean_distance(moved, target)
        for candidate in itertools.permutations(list(source), 3):
            other, _, _ = geometry.init_icp(source, target, candidate, False)
            assert geometry.mean_distance(other, target) >= best


def test_the_pair_cache_returns_the_same_alignment_every_time():
    """`_TargetPairs` answers from a dict after the first call; a cached None
    (a target pair that cannot be scaled) must come back as None, not as a
    cache miss recomputed."""
    target = {"A": np.zeros(3), "B": np.zeros(3), "C": np.array([1.0, 2.0, 3.0])}
    pairs = geometry._TargetPairs(target)
    assert pairs("A", "B") is None
    assert pairs("A", "B") is None
    first = pairs("A", "C")
    assert first is pairs("A", "C")
