import numpy as np

from typing import Tuple, Any

from joblib import Memory
from nilearn import datasets, masking
from scipy.spatial import distance_matrix

from fugw.scripts import coarse_to_fine, lmds


# Create a Memory object to handle the caching
fugw_data = "~/fugw_data"
memory = Memory(fugw_data, verbose=0)


def _check_masker(mask: str) -> None:
    """Check if the mask is valid."""
    valid_masks = ["mni152_gm_mask", "mni152_brain_mask"]
    if mask not in valid_masks:
        raise ValueError(
            f"Unknown mask {mask}. Valid masks include {valid_masks}."
        )


def _compute_connected_segmentation(mask_img: Any) -> np.ndarray:
    """Compute the connected segmentation of the mask from a 3D Nifti image."""
    return (
        masking.compute_background_mask(mask_img, connected=True).get_fdata()
        > 0
    )


@memory.cache
def _fetch_geometry_full_rank(
    mask: str, resolution: int, method: str = "euclidean"
) -> Tuple[np.ndarray, float]:
    """Returns the normalized full-rank distance matrix for the
    given mesh and the maximum distance between two points in the volume.
    """
    if mask == "mni152_gm_mask":
        mask_img = datasets.load_mni152_gm_mask(resolution=resolution)
    elif mask == "mni152_brain_mask":
        mask_img = datasets.load_mni152_brain_mask(resolution=resolution)
    segmentation = _compute_connected_segmentation(mask_img)

    if method == "geodesic":
        raise NotImplementedError(
            "Geodesic distance computation is not implemented for volume data"
            " in the full-rank setting."
        )

    elif method == "euclidean":
        # Return euclidean distance matrix
        coordinates = np.array(np.where(segmentation)).T
        geometry = distance_matrix(coordinates, coordinates)

    # Normalize the distance matrix
    d_max = geometry.max()
    geometry = geometry / d_max

    return geometry, d_max


@memory.cache
def _fetch_geometry_low_rank(
    mask: str,
    resolution: str,
    method: str = "euclidean",
    rank: int = 3,
    n_landmarks: int = 100,
    n_jobs: int = 2,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """Returns the normalized low-rank distance matrix for the
    given mesh and the maximum distance between two points in the mesh.
    """
    if mask == "mni152_gm_mask":
        mask_img = datasets.load_mni152_gm_mask(resolution=resolution)
    elif mask == "mni152_brain_mask":
        mask_img = datasets.load_mni152_brain_mask(resolution=resolution)
    segmentation = _compute_connected_segmentation(mask_img)

    # Get the anisotropy of the 3D mask
    anisotropy = np.abs(mask_img.header.get_zooms()[:3]).tolist()

    geometry_embedding = lmds.compute_lmds_volume(
        segmentation,
        method=method,
        k=rank,
        n_landmarks=n_landmarks,
        anisotropy=anisotropy,
        n_jobs=n_jobs,
        verbose=verbose,
    ).nan_to_num()

    (
        geometry_embedding_normalized,
        d_max,
    ) = coarse_to_fine.random_normalizing(geometry_embedding)

    return (
        geometry_embedding_normalized.cpu().numpy(),
        d_max,
    )


def fetch_vol_geometry(
    mask: str,
    resolution: int,
    method: str = "euclidean",
    rank: int = -1,
    n_landmarks: int = 100,
    n_jobs: int = 2,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """Returns either the normalized full-rank or low-rank embedding
    of the distance matrix for the given mesh and the maximum distance
    between two points in the volume.

    Parameters
    ----------
    mask : str
        Input mask name. Valid masks include "mni152_gm_mask" and
        "mni152_brain_mask".
    resolution : int
        Input resolution name.
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
        distance encountered in the volume.
    """
    _check_masker(mask)

    if rank == -1:
        return _fetch_geometry_full_rank(
            mask=mask, resolution=resolution, method=method
        )
    else:
        return _fetch_geometry_low_rank(
            mask=mask,
            resolution=resolution,
            method=method,
            rank=rank,
            n_landmarks=n_landmarks,
            n_jobs=n_jobs,
            verbose=verbose,
        )
