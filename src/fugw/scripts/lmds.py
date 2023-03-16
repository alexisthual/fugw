import contextlib
import warnings

import gdist
import joblib
import networkx as nx
import numpy as np
import torch

from fugw.utils import get_progress
from joblib import delayed, Parallel
from scipy.sparse import coo_matrix


@contextlib.contextmanager
def rich_progress_joblib(description=None, total=None, verbose=False):
    if description is None:
        description = "Processing..."

    progress = get_progress()
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


def compute_geodesic_distances(coordinates, triangles, index):
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


def adjacency_matrix_from_triangles(n, triangles):
    edges = np.hstack(
        (
            np.vstack((triangles[:, 0], triangles[:, 1])),
            np.vstack((triangles[:, 0], triangles[:, 2])),
            np.vstack((triangles[:, 1], triangles[:, 0])),
            np.vstack((triangles[:, 1], triangles[:, 2])),
            np.vstack((triangles[:, 2], triangles[:, 0])),
            np.vstack((triangles[:, 2], triangles[:, 1])),
        )
    )
    values = np.ones(edges.shape[1])

    # Divide data by 2 since all edges i -> j are counted twice
    # because they all belong to exactly two triangles on the mesh
    adjacency = coo_matrix((values, edges), (n, n)).tocsr() / 2

    # Making it symmetrical
    adjacency = (adjacency + adjacency.T) / 2

    return adjacency


def compute_geodesic_distances_from_adjacency(coordinates, adjacency, index):
    def weights(u, v, _):
        return np.linalg.norm(coordinates[u] - coordinates[v])

    G = nx.Graph(adjacency)
    d = nx.single_source_dijkstra_path_length(G, index, weight=weights)
    geodesic_distances = np.array(list(d.values()))[list(d.keys())]

    return torch.from_numpy(geodesic_distances)


def compute_lmds(
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
        "Geodesic_distances for landmarks",
        total=basis_indices.shape[0],
        verbose=verbose,
    ):
        basis_distance = torch.vstack(
            Parallel(n_jobs=n_jobs)(
                delayed(compute_geodesic_distances)(
                    coordinates,
                    triangles,
                    index,
                )
                for index in basis_indices
            )
        )[:, indices]
    basis_distance = basis_distance.type(torch.float32)

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
