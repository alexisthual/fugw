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
    valid_masks = ["mni152_gm_mask", "mni152_brain_mask"]
    if mask not in valid_masks:
        raise ValueError(
            f"Unknown mask {mask}. Valid masks include {valid_masks}."
        )


def _compute_connected_segmentation(mask_img: Any) -> np.ndarray:
    return (
        masking.compute_background_mask(mask_img, connected=True).get_fdata()
        > 0
    )


@memory.cache
def _fetch_geometry_full_rank(
    mask: str, resolution: int, method: str = "euclidean"
):
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
):
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
