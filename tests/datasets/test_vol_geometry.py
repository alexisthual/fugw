import numpy as np
import pytest

from fugw.datasets.vol_geometry import (
    fetch_vol_geometry,
    _check_masker,
    _fetch_geometry_full_rank,
    _fetch_geometry_low_rank,
)

valid_masks = [
    "mni152_gm_mask",
    "mni152_brain_mask",
]


def test_check_mesh():
    valid_mask = valid_masks[0]
    invalid_mask = "mni200"

    assert _check_masker(valid_mask) is None
    with pytest.raises(ValueError):
        _check_masker(invalid_mask)


def test_fetch_geometry_full_rank():
    mask = "mni152_gm_mask"
    resolution = 10
    n_voxels = 1886  # Number of voxels in the mask
    geometry, d_max = _fetch_geometry_full_rank(
        mask, resolution, method="euclidean"
    )

    assert isinstance(d_max, float)
    assert isinstance(geometry, np.ndarray)
    assert geometry.shape == (n_voxels, n_voxels)


@pytest.mark.parametrize(
    "method",
    ["geodesic", "euclidean"],
)
def test_fetch_geometry_low_rank(method):
    mask = "mni152_gm_mask"
    resolution = 10
    n_voxels = 1886  # Number of voxels in the mask
    rank = 3
    geometry, d_max = _fetch_geometry_low_rank(
        mask, resolution, method, rank, verbose=False
    )

    assert isinstance(d_max, float)
    assert isinstance(geometry, np.ndarray)
    assert geometry.shape == (n_voxels, rank)


def test_fetch_vol_geometry():
    mask = "mni152_gm_mask"
    resolution = 10
    n_voxels = 1886  # Number of voxels in the mask

    rank = -1
    geometry, d_max = fetch_vol_geometry(mask, resolution, rank=rank)

    assert isinstance(d_max, float)
    assert isinstance(geometry, np.ndarray)
    assert geometry.shape == (n_voxels, n_voxels)

    rank = 3
    geometry, d_max = fetch_vol_geometry(
        mask, resolution, rank=rank, verbose=False
    )

    assert isinstance(d_max, float)
    assert isinstance(geometry, np.ndarray)
    assert geometry.shape == (n_voxels, rank)
