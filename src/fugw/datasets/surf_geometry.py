import gdist
import numpy as np

from typing import Tuple

from joblib import Memory
from nilearn import surface, datasets
from scipy.spatial import distance_matrix

from fugw.scripts import coarse_to_fine, lmds


# Create a Memory object to handle the caching
fugw_data = "~/fugw_data"
memory = Memory(fugw_data, verbose=0)


def _check_mesh(mesh: str) -> None:
    """Check if the mesh is valid."""
    valid_meshes = ["pial_left", "pial_right", "infl_left", "infl_right"]
    if mesh not in valid_meshes:
        raise ValueError(
            f"Unknown mesh {mesh}. Valid meshes include {valid_meshes}."
        )


def _check_resolution(resolution: str) -> None:
    """Check if the resolution is valid."""
    valid_resolutions = [
        "fsaverage3",
        "fsaverage4",
        "fsaverage5",
        "fsaverage6",
        "fsaverage7",
        "fsaverage",
    ]
    if resolution not in valid_resolutions:
        raise ValueError(
            f"Unknown resolution {resolution}. Valid resolutions include"
            f" {valid_resolutions}."
        )


@memory.cache
def _fetch_geometry_full_rank(
    mesh: str, resolution: str, method: str = "geodesic"
) -> Tuple[np.ndarray, float]:
    """Returns the normalized full-rank distance matrix for the
    given mesh and the maximum distance between two points in the mesh.
    """
    mesh_path = datasets.fetch_surf_fsaverage(mesh=resolution)[mesh]
    (coordinates, triangles) = surface.load_surf_mesh(mesh_path)
    if method == "geodesic":
        # Return geodesic distance matrix
        geometry = gdist.local_gdist_matrix(
            coordinates.astype(np.float64), triangles.astype(np.int32)
        ).toarray()

    elif method == "euclidean":
        # Return euclidean distance matrix
        geometry = distance_matrix(coordinates, coordinates)

    # Normalize the distance matrix
    d_max = geometry.max()
    geometry = geometry / d_max

    return geometry, d_max


@memory.cache
def _fetch_geometry_low_rank(
    mesh: str,
    resolution: str,
    method: str = "geodesic",
    rank: int = 3,
    n_landmarks: int = 100,
    n_jobs: int = 2,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """Returns the normalized low-rank distance matrix for the
    given mesh and the maximum distance between two points in the mesh.
    """
    if method == "euclidean":
        raise NotImplementedError(
            "Low-rank embedding is not implemented for L2 distance matrices."
        )

    mesh_path = datasets.fetch_surf_fsaverage(mesh=resolution)[mesh]
    (coordinates, triangles) = surface.load_surf_mesh(mesh_path)
    geometry_embedding = lmds.compute_lmds_mesh(
        coordinates,
        triangles,
        n_landmarks=n_landmarks,
        k=rank,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    (
        geometry_embedding_normalized,
        d_max,
    ) = coarse_to_fine.random_normalizing(geometry_embedding)

    return geometry_embedding_normalized.cpu().numpy(), d_max


def fetch_surf_geometry(
    mesh: str,
    resolution: str,
    method: str = "geodesic",
    rank: int = -1,
    n_landmarks: int = 100,
    n_jobs: int = 2,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """Returns either the normalized full-rank or low-rank embedding
    of the distance matrix for the given mesh and the maximum distance
    between two points in the mesh.

    Parameters
    ----------
    mesh : str
        Input mesh name. Valid meshes include "pial_left", "pial_right",
        "infl_left", and "infl_right".
    resolution : str
        Input resolution name. Valid resolutions include "fsaverage3",
        "fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7", and
        "fsaverage".
    method : str, optional
        Method used to compute distances, either "geodesic" or "euclidean",
        by default "geodesic".
    rank : int, optional
        Dimension of embedding, -1 for full-rank embedding
        and rank < n_vertices for low-rank embedding, by default -1
    n_landmarks : int, optional
        Number of vertices to sample on mesh to approximate embedding,
        by default 100
    n_jobs : int, optional,
        Relative tolerance used to check intermediate results, by default 2
    verbose : bool, optional
        Enable logging, by default True

    Returns
    -------
    Tuple[np.ndarray, float]
        Full-rank or low-rank embedding of the distance matrix of size
        (n_vertices, n_vertices) or (n_vertices, rank) and the maximum
        distance encountered in the mesh.
    """
    _check_mesh(mesh)
    _check_resolution(resolution)

    if rank == -1:
        return _fetch_geometry_full_rank(
            mesh=mesh, resolution=resolution, method=method
        )
    else:
        return _fetch_geometry_low_rank(
            mesh=mesh,
            resolution=resolution,
            method=method,
            rank=rank,
            n_landmarks=n_landmarks,
            n_jobs=n_jobs,
            verbose=verbose,
        )
