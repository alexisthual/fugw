import contextlib
import warnings

import gdist
import joblib
import numpy as np
import torch

from fugw.utils import _get_progress
from joblib import delayed, Parallel
from dijkstra3d import euclidean_distance_field


@contextlib.contextmanager
def rich_progress_joblib(description=None, total=None, verbose=False):
    if description is None:
        description = "Processing..."

    progress = _get_progress()
    if verbose:
        task_id = progress.add_task(description, total=total)

    class BatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            if verbose:
                progress.update(task_id, advance=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack

    try:
        joblib.parallel.BatchCompletionCallBack = BatchCompletionCallback
        progress.start()

        yield progress
    finally:
        progress.stop()
        joblib.parallel.BatchCompletionCallBack = old_callback


def compute_gdist(coordinates, triangles, index):
    if isinstance(coordinates, torch.Tensor) or isinstance(
        triangles, torch.Tensor
    ):
        coordinates = np.array(coordinates)
        triangles = np.array(triangles)

    geodesic_distances = torch.from_numpy(
        gdist.compute_gdist(
            coordinates.astype(np.float64),
            triangles.astype(np.int32),
            np.array([index]).astype(np.int32),
        )
    )

    return geodesic_distances


def compute_distance_field(coordinates):
    """Create a binary mask from array of integer coordinates."""
    mask = np.zeros(coordinates.max(axis=0) + 1, dtype=np.uint8)
    mask[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1

    return mask


def compute_geodesic_distances_from_volume(
    field, coordinates, index, anisotropy=(1, 1, 1)
):
    """
    Computes the list of geodesic distances from a given index
    in the point cloud.

    Parameters
    ----------
        field: ndarray of size (x, y, z)
            3D binary mask of the region of interest
        coordinates: ndarray of size (n, 3)
            Array of 3D coordinates of the point cloud
        index: int
            Index of the source point in the point cloud
        anisotropy: tuple, optional, defaults to (1, 1, 1).
            (x, y, z)-anisotropy of the voxels

    Returns:
    --------
        torch.Tensor of size (n,)
            Tensor of geodesic distances from the source point
    """
    source = coordinates[index]

    # Compute the distance field
    distance_field = euclidean_distance_field(
        field, source=source, anisotropy=anisotropy
    )

    # Retrieve only the non-infinite distances
    dists = distance_field[
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    ]
    return torch.from_numpy(dists).to(torch.float64)


def compute_euclidean_distance(coordinates, index):
    """Compute pairwise euclidean distance between coordinates."""
    return (
        torch.cdist(coordinates, coordinates[index].unsqueeze(0))
        .flatten()
        .to(torch.float64)
    )


def _compute_lmds(
    basis_distance,
    n_landmarks,
    n_voxels,
    k,
    tol,
    invert_indices,
):
    """Compute LMDS embedding from precomputed geodesic
    distances between landmarks.

    This implementation follows a method described in
    Platt, John. ‘FastMap, MetricMap, and Landmark MDS
    Are All Nystrom Algorithms’, 1 January 2005.
    https://www.microsoft.com/en-us/research/publication/
    fastmap-metricmap-and-landmark-mds-are-all-nystrom-algorithms/.
    """
    E = basis_distance[:, :n_landmarks]
    F = basis_distance[:, n_landmarks:]

    E_squared = E * E
    E_squared_sum = E_squared.sum()

    F_squared = F * F

    # Check E is symmetric
    E_Et_abs_max = torch.max(torch.abs(E - E.T))
    E_abs_max = torch.max(torch.abs(E))
    if not E_Et_abs_max <= tol * E_abs_max:
        warnings.warn(
            f"E might not be symmetric ({E_Et_abs_max} > {E_abs_max})"
        )

    # Double centring of A
    A = (
        -(
            E_squared
            - (
                torch.tile(torch.sum(E_squared, dim=0), (n_landmarks, 1))
                / n_landmarks
            )
            - (
                torch.tile(torch.sum(E_squared, dim=1), (n_landmarks, 1)).T
                / n_landmarks
            )
            + E_squared_sum / (n_landmarks**2)
        )
        / 2
    )

    # Check A is symmetric
    A_At_abs_max = torch.max(torch.abs(A - A.T))
    A_abs_max = torch.max(torch.abs(A))
    if not A_At_abs_max <= tol * A_abs_max:
        warnings.warn(
            f"A might not be symmetric ({A_At_abs_max} > {A_abs_max})"
        )

    # Check that np.ones(n_landmarks) is an eigen vector
    # associated with eigen value 0 to control double centering
    t = torch.ones(n_landmarks) @ A @ torch.ones(n_landmarks)
    if not t <= tol:
        warnings.warn(f"A might not be centered ({t} > {tol})")

    B = (
        -1
        / 2
        * (
            F_squared
            - torch.tile(
                torch.sum(E_squared, axis=1), (n_voxels - n_landmarks, 1)
            ).T
            / n_landmarks
        )
    )

    S, U = torch.linalg.eig(A)
    S = S.real
    U = U.real

    eig_indices = S.argsort().flip(0)
    S_min = torch.min(S)
    S = S[eig_indices] - S_min
    U = U[:, eig_indices]

    X = torch.vstack(
        [
            (U * torch.sqrt(torch.abs(S)))[:, :k],
            (U[:, :k].T @ B).T
            / torch.tile(
                torch.sqrt(torch.abs(S))[:k], (n_voxels - n_landmarks, 1)
            ),
        ]
    )

    # Reorder voxels to match original mesh order
    X = X[invert_indices]

    return X


def compute_lmds_mesh(
    coordinates,
    triangles,
    n_landmarks=100,
    k=3,
    n_jobs=2,
    tol=1e-3,
    verbose=False,
):
    """
    Compute embedding in k-dimension approximating
    the matrix D of geodesic distances on a given mesh

    Parameters
    ----------
    coordinates: array of size (n, 3)
        Coordinates of vertices
    triangles: array of size (t, 3)
        Triplets of indices indicating faces
    n_landmarks: int, optional, defaults to 100
        Number of vertices to sample on mesh to approximate embedding
    k: int, optional, defaults to 3
        Dimension of embedding
    n_jobs: int, optional, defaults to 2
        Number of CPUs to use to parallelise computation
    tol: float, optional, defaults to 1e-3
        Relative tolerance used to check intermediate results
    verbose: bool, optional, defaults to False
        Log solving process

    Returns
    -------
    X: torch.Tensor of size (n, k)
        Embedding such that cdist(X, X) approximates D
    """
    n_voxels = coordinates.shape[0]
    indices = torch.randperm(n_voxels)
    invert_indices = torch.empty_like(indices)
    invert_indices[indices] = torch.arange(n_voxels)
    basis_indices = indices[:n_landmarks]

    with rich_progress_joblib(
        "Geodesic distances (landmarks)",
        total=basis_indices.shape[0],
        verbose=verbose,
    ):
        basis_distance = torch.vstack(
            Parallel(n_jobs=n_jobs)(
                delayed(compute_gdist)(
                    coordinates,
                    triangles,
                    index,
                )
                for index in basis_indices
            )
        )[:, indices]
    basis_distance = basis_distance.type(torch.float32)

    return _compute_lmds(
        basis_distance, n_landmarks, n_voxels, k, tol, invert_indices
    )


def compute_lmds_volume(
    segmentation,
    method="geodesic",
    anisotropy=(1, 1, 1),
    n_landmarks=100,
    k=3,
    n_jobs=2,
    tol=1e-3,
    verbose=False,
):
    """
    Compute embedding in k-dimension approximating
    the matrix D of geodesic distances on a given volume

    Parameters
    ----------
    segmentation: ndarray of size (n, 3)
        Binary mask of the ROI
    anisotropy: tuple of size 3, optional, defaults to (1,1,1)
        Anisotropy of the voxels
    method: str, optional, defaults to "geodesic"
        Method used to compute distances, either "geodesic" or "euclidean"
    n_landmarks: int, optional, defaults to 100
        Number of vertices to sample on mesh to approximate embedding
    k: int, optional, defaults to 3
        Dimension of embedding
    n_jobs: int, optional, defaults to 2
        Number of CPUs to use to parallelise computation
    tol: float, optional, defaults to 1e-3
        Relative tolerance used to check intermediate results
    verbose: bool, optional, defaults to False
        Log solving process

    Returns
    -------
    X: torch.Tensor of size (n, k)
        Embedding such that cdist(X, X) approximates D
    """

    coordinates = np.array(np.nonzero(segmentation)).T
    field = compute_distance_field(coordinates)
    n_voxels = coordinates.shape[0]
    indices = torch.randperm(n_voxels)
    invert_indices = torch.empty_like(indices)
    invert_indices[indices] = torch.arange(n_voxels)
    basis_indices = indices[:n_landmarks]

    with rich_progress_joblib(
        "Geodesic distances (landmarks)",
        total=basis_indices.shape[0],
        verbose=verbose,
    ):
        if method == "geodesic":
            basis_distance = torch.vstack(
                Parallel(n_jobs=n_jobs)(
                    delayed(compute_geodesic_distances_from_volume)(
                        field,
                        coordinates,
                        index,
                        anisotropy,
                    )
                    for index in basis_indices
                )
            )[:, indices]

        elif method == "euclidean":
            basis_distance = torch.vstack(
                Parallel(n_jobs=n_jobs)(
                    delayed(compute_euclidean_distance)(
                        torch.from_numpy(coordinates).to(torch.float64),
                        index,
                    )
                    for index in basis_indices
                )
            )[:, indices]

    basis_distance = basis_distance.type(torch.float32)

    return _compute_lmds(
        basis_distance, n_landmarks, n_voxels, k, tol, invert_indices
    )
