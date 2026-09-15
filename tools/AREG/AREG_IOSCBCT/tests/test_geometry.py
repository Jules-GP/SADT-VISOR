"""The registration arithmetic: a closed-form landmark fit, then an ICP.

Neither needs a card, a checkpoint or another tool, so both run for real here.
Two stages, and the order is load-bearing: the landmark transform puts the two
meshes in roughly the same place so the ICP starts inside its capture range.

The ICP's arithmetic is kept exactly as upstream wrote it -- this is a
repackaging, and swapping the estimator would change every result the tool has
produced. Only its NAME was corrected: upstream calls it point-to-plane and
computes point-to-point.
"""

import numpy as np
import pytest

from sadt_areg_ioscbct import geometry


def rigid_matrix(angle=0.21, translation=(4.0, -3.0, 2.5)):
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


CROWNS = np.array([
    [0.0, 0.0, 0.0], [10.0, 0.0, 1.0], [20.0, 4.0, 0.0],
    [0.0, 10.0, 2.0], [10.0, 10.0, 0.0], [20.0, 14.0, 3.0],
])


# ---------------------------------------------------------------------------
# align_by_landmarks
# ---------------------------------------------------------------------------

def test_a_known_rigid_motion_is_recovered_to_the_float():
    """Closed form, not a search: same landmarks in, same matrix out, every
    time. The whole two-stage design rests on that."""
    truth = rigid_matrix()
    matrix = geometry.align_by_landmarks(CROWNS, geometry.apply(CROWNS, truth))
    assert matrix == pytest.approx(truth, abs=1e-6)


def test_the_fit_is_rigid_even_when_the_points_do_not_agree():
    """RigidBody mode, so no scaling and no reflection can sneak in through a
    noisy landmark -- a scaled intraoral scan is a wrong measurement that looks
    like a good registration."""
    rng = np.random.default_rng(3)
    noisy = geometry.apply(CROWNS, rigid_matrix()) + rng.normal(scale=0.8, size=CROWNS.shape)

    rotation = geometry.align_by_landmarks(CROWNS, noisy)[:3, :3]
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)


def test_the_same_landmarks_always_give_the_same_matrix():
    """Two runs of one request on one dataset must agree: this is patient data
    being resampled."""
    target = geometry.apply(CROWNS, rigid_matrix())
    first = geometry.align_by_landmarks(CROWNS, target)
    second = geometry.align_by_landmarks(CROWNS, target)
    assert np.array_equal(first, second)


def test_mismatched_point_counts_are_refused_with_both_counts():
    with pytest.raises(ValueError) as raised:
        geometry.align_by_landmarks(CROWNS, CROWNS[:4])
    assert "6 moving against 4 fixed" in str(raised.value)


def test_fewer_than_three_points_is_refused_with_the_count():
    """Two points fix a line, not a frame; the fit would return something and
    it would be wrong about the rotation around that line."""
    with pytest.raises(ValueError) as raised:
        geometry.align_by_landmarks(CROWNS[:2], CROWNS[:2])
    assert "at least 3 shared points, got 2" in str(raised.value)


def test_exactly_three_points_is_accepted():
    truth = rigid_matrix()
    matrix = geometry.align_by_landmarks(CROWNS[:3], geometry.apply(CROWNS[:3], truth))
    assert matrix == pytest.approx(truth, abs=1e-6)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def test_apply_rotates_then_translates():
    matrix = rigid_matrix(angle=0.0, translation=(1.0, 2.0, 3.0))
    assert geometry.apply(np.array([[0.0, 0.0, 0.0]]), matrix)[0] == pytest.approx([1, 2, 3])


def test_apply_leaves_its_input_alone():
    """It returns a new array: the caller writes the ORIGINAL mesh's points
    back out under a different transform in the same loop."""
    points = CROWNS.copy()
    geometry.apply(points, rigid_matrix())
    assert np.array_equal(points, CROWNS)


def test_apply_accepts_a_list_of_tuples():
    """vtk hands back tuples, and `@` on a list of tuples is a TypeError."""
    assert geometry.apply([(1.0, 0.0, 0.0)], np.eye(4))[0] == pytest.approx([1, 0, 0])


# ---------------------------------------------------------------------------
# icp_point_to_point
# ---------------------------------------------------------------------------

def _cloud(count=400, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=6.0, size=(count, 3))


def test_the_icp_recovers_a_small_displacement():
    fixed = _cloud()
    displacement = rigid_matrix(angle=0.02, translation=(0.4, -0.3, 0.2))
    moving = geometry.apply(fixed, np.linalg.inv(displacement))

    matrix, stats = geometry.icp_point_to_point(moving, fixed, max_dist=5.0)
    assert geometry.apply(moving, matrix) == pytest.approx(fixed, abs=1e-3)
    assert stats["rmse"] < 1e-3
    assert stats["fitness"] == pytest.approx(1.0)
    assert stats["iterations"] >= 1


def test_the_icp_reports_what_it_did_rather_than_only_that_it_returned():
    """A caller has to be able to say whether the registration converged. The
    three numbers travel into the run report."""
    fixed = _cloud()
    _matrix, stats = geometry.icp_point_to_point(fixed, fixed, max_dist=1.5)
    assert set(stats) == {"rmse", "fitness", "iterations"}


def test_a_max_dist_too_small_to_match_anything_stops_rather_than_wandering():
    """Below the point spacing nothing is a correspondence, and continuing on
    fewer than three would be fitting a frame to a line."""
    fixed = _cloud()
    moving = geometry.apply(fixed, rigid_matrix(angle=0.5, translation=(50.0, 0.0, 0.0)))

    matrix, stats = geometry.icp_point_to_point(moving, fixed, max_dist=1e-6)
    assert np.array_equal(matrix, np.eye(4))
    assert stats["fitness"] == 0.0


def test_the_icp_never_returns_a_reflection():
    """Flipping the smallest singular vector is the standard repair. Without
    it a mesh comes back MIRRORED with a perfectly good RMSE -- a left canine
    where the right one should be.

    Built so the correspondence really is the mirror pairing: a near-flat
    sheet, mirrored through its own plane, so each moved point's nearest
    neighbour is its own reflection and the covariance is genuinely
    orientation-reversing. On a round cloud the nearest neighbours are a
    scramble and the SVD never sees the reflection at all.
    """
    rng = np.random.default_rng(5)
    grid = np.array([[float(x), float(y), 0.0] for x in range(18) for y in range(18)])
    grid[:, 2] = rng.uniform(-0.04, 0.04, size=len(grid))
    moving = grid * np.array([1.0, 1.0, -1.0])

    matrix, _stats = geometry.icp_point_to_point(moving, grid, max_dist=0.5)
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0, abs=1e-6)


def test_the_icp_needs_three_points_on_each_side():
    cloud = _cloud(count=10)
    with pytest.raises(ValueError, match="at least 3 points"):
        geometry.icp_point_to_point(cloud[:2], cloud)
    with pytest.raises(ValueError, match="at least 3 points"):
        geometry.icp_point_to_point(cloud, cloud[:2])


def test_the_upstream_loop_bounds_are_kept():
    """The thresholds decide when the registration stops moving, so changing
    one changes results. Pinned rather than tuned."""
    assert geometry.DEFAULT_MAX_DIST == 1.5
    assert geometry._MAX_ITERATIONS == 2000
    assert geometry._RMSE_THRESHOLD == 1e-8
    assert geometry._FITNESS_THRESHOLD == 1e-8


def test_the_icp_stops_when_it_stops_improving_rather_than_running_out():
    """2000 iterations on a real arch is minutes. Two identical clouds
    converge on the second pass."""
    fixed = _cloud()
    _matrix, stats = geometry.icp_point_to_point(fixed, fixed, max_dist=1.5)
    assert stats["iterations"] < 5
