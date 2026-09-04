"""The registration itself: landmarks first, then a point-to-plane ICP.

Ported from upstream's `AREG_IOSCBCT/AREG_IOSCBCT.py`, which is a Slicer CLI
module. Nothing here predicts anything -- the landmarks, the tooth labels and
the orientation all arrive as inputs, produced by other tools. That is what
lets this tool depend on neither torch nor pytorch3d while driving engines
pinned to both.

Two stages, and the order is load-bearing: the landmark transform puts the two
meshes in roughly the same place so the ICP starts inside its capture range. An
ICP started on unaligned meshes converges to whatever local minimum it happens
to reach.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# How far a point may be from its nearest neighbour and still count as a
# correspondence, in millimetres. Upstream's value.
DEFAULT_MAX_DIST = 1.5

# Upstream's loop bounds, kept as they are: the thresholds decide when the
# registration stops moving, and changing them changes results.
_MAX_ITERATIONS = 2000
_RMSE_THRESHOLD = 1e-8
_FITNESS_THRESHOLD = 1e-8


def align_by_landmarks(moving_lms, fixed_lms) -> np.ndarray:
    """The 4x4 that best maps `moving_lms` onto `fixed_lms`, rigid.

    `vtkLandmarkTransform` in RigidBody mode, which is a closed-form fit rather
    than a search: same landmarks in, same matrix out, every time.

    It takes the landmarks and nothing else. It used to take the mesh's points
    as its first argument and never read them, which reads as though the fit
    were influenced by the surface -- it is not, and that is the property the
    two-stage design depends on.
    """
    import vtk

    if len(moving_lms) != len(fixed_lms):
        raise ValueError(
            "Landmark alignment needs the same points on both sides: "
            f"{len(moving_lms)} moving against {len(fixed_lms)} fixed."
        )
    if len(moving_lms) < 3:
        raise ValueError(
            f"Landmark alignment needs at least 3 shared points, got {len(moving_lms)}."
        )

    source, target = vtk.vtkPoints(), vtk.vtkPoints()
    for point in moving_lms:
        source.InsertNextPoint(*point)
    for point in fixed_lms:
        target.InsertNextPoint(*point)

    transform = vtk.vtkLandmarkTransform()
    transform.SetSourceLandmarks(source)
    transform.SetTargetLandmarks(target)
    transform.SetModeToRigidBody()
    transform.Update()

    matrix = np.eye(4)
    vtk_matrix = transform.GetMatrix()
    for row in range(4):
        for column in range(4):
            matrix[row, column] = vtk_matrix.GetElement(row, column)
    return matrix


def icp_point_to_point(moving_points: np.ndarray, fixed_points: np.ndarray,
                       max_dist: float = DEFAULT_MAX_DIST) -> tuple:
    """Refine an alignment; return `(4x4 matrix, {rmse, fitness, iterations})`.

    **Renamed from upstream's `run_icp_point_to_plane`, which is not what it
    computes.** The update below is the point-to-point SVD -- centre both point
    sets, take the SVD of their covariance, repair a reflection. A true
    point-to-plane step minimises distance along the fixed surface's NORMALS and
    solves a linearised 6x6 system; upstream computes the fixed mesh's normals
    and then never uses them.

    The arithmetic is kept exactly as upstream wrote it: this is a repackaging,
    and swapping the estimator would change every result the tool has produced.
    Only the name is corrected, which changes nothing and stops the next reader
    trusting a label that contradicts the code under it.

    Returns the metrics as well as the matrix, so a caller can report whether
    the registration actually converged rather than only that it returned.
    """
    from scipy.spatial import cKDTree

    if len(moving_points) < 3 or len(fixed_points) < 3:
        raise ValueError("ICP needs at least 3 points on each mesh.")

    transformation = np.eye(4)
    current = np.asarray(moving_points, dtype=float).copy()
    tree = cKDTree(np.asarray(fixed_points, dtype=float))

    previous_rmse, previous_fitness = np.inf, 0.0
    rmse, fitness, iteration = np.inf, 0.0, 0

    for iteration in range(_MAX_ITERATIONS):
        distances, indices = tree.query(current, k=1)
        valid = distances < max_dist
        if valid.sum() < 3:
            logger.warning("ICP stopped at iteration %d: fewer than 3 correspondences", iteration)
            break

        rmse = float(np.sqrt(np.mean(distances[valid] ** 2)))
        fitness = float(valid.sum() / len(current))
        if (abs(previous_rmse - rmse) < _RMSE_THRESHOLD
                and abs(previous_fitness - fitness) < _FITNESS_THRESHOLD):
            break
        previous_rmse, previous_fitness = rmse, fitness

        source = current[valid]
        target = np.asarray(fixed_points, dtype=float)[indices[valid]]
        source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
        covariance = (source - source_centre).T @ (target - target_centre)
        u, _s, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            # A reflection is not a rigid motion: flipping the smallest singular
            # vector is the standard repair, and without it a mesh can come back
            # mirrored with a perfectly good RMSE.
            vt[-1, :] *= -1
            rotation = vt.T @ u.T
        translation = target_centre - rotation @ source_centre

        step = np.eye(4)
        step[:3, :3] = rotation
        step[:3, 3] = translation
        transformation = step @ transformation
        current = (rotation @ current.T).T + translation

    return transformation, {
        "rmse": rmse,
        "fitness": fitness,
        "iterations": iteration + 1,
    }


def apply(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """`points` through a 4x4, as a new array."""
    points = np.asarray(points, dtype=float)
    return (matrix[:3, :3] @ points.T).T + matrix[:3, 3]
