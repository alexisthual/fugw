from itertools import product

import numpy as np
import pytest
import torch

from fugw.scripts import coarse_to_fine
from fugw.mappings import FUGW, FUGWSparse
from fugw.utils import _init_mock_distribution
from nilearn import datasets, surface

np.random.seed(0)
torch.manual_seed(0)

n_voxels_source = 105
n_samples_source = 50
n_voxels_target = 95
n_samples_target = 45
n_features_train = 10
n_features_test = 5

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))

return_numpys = [False, True]


@pytest.mark.parametrize("return_numpy", product(return_numpys))
def test_random_normalizing(return_numpy):
    _, _, _, embeddings = _init_mock_distribution(
        n_features_train, n_voxels_source, return_numpy=return_numpy
    )

    embeddings_normalized, d_max = coarse_to_fine.random_normalizing(
        embeddings
    )
    assert isinstance(d_max, float)
    assert embeddings_normalized.shape == embeddings.shape


def test_uniform_mesh_sampling():
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage4")
    coordinates, triangles = surface.load_surf_mesh(fsaverage.pial_left)

    n_samples = 104
    sample = coarse_to_fine.sample_mesh_uniformly(
        coordinates, triangles, n_samples=n_samples
    )

    assert sample.shape == (n_samples,)
    # All sampled indices should be different
    assert np.unique(sample).shape == (n_samples,)


@pytest.mark.parametrize(
    "device,return_numpy", product(devices, return_numpys)
)
def test_coarse_to_fine(device, return_numpy):
    _, source_features, _, source_embeddings = _init_mock_distribution(
        n_features_train, n_voxels_source, return_numpy=return_numpy
    )
    _, target_features, _, target_embeddings = _init_mock_distribution(
        n_features_train, n_voxels_target, return_numpy=return_numpy
    )

    coarse_mapping = FUGW()
    coarse_mapping_solver = "mm"

    fine_mapping = FUGWSparse()
    fine_mapping_solver = "mm"

    # Sub-sample source and target distributions
    source_sample = torch.randperm(n_voxels_source)[:n_samples_source]
    target_sample = torch.randperm(n_voxels_target)[:n_samples_target]

    source_sample, target_sample = coarse_to_fine.fit(
        coarse_mapping=coarse_mapping,
        coarse_mapping_solver=coarse_mapping_solver,
        fine_mapping=fine_mapping,
        fine_mapping_solver=fine_mapping_solver,
        source_sample=source_sample,
        target_sample=target_sample,
        source_features=source_features,
        target_features=target_features,
        source_geometry_embeddings=source_embeddings,
        target_geometry_embeddings=target_embeddings,
        device=device,
    )

    assert coarse_mapping.pi.shape == (n_samples_source, n_samples_target)
    assert fine_mapping.pi.shape == (n_voxels_source, n_voxels_target)

    # Use trained model to transport new features
    # 1. with numpy arrays
    source_features_test = np.random.rand(n_features_test, n_voxels_source)
    target_features_test = np.random.rand(n_features_test, n_voxels_target)
    source_features_on_target = fine_mapping.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, np.ndarray)
    target_features_on_source = fine_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, np.ndarray)

    source_features_test = np.random.rand(n_voxels_source)
    target_features_test = np.random.rand(n_voxels_target)
    source_features_on_target = fine_mapping.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, np.ndarray)
    target_features_on_source = fine_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, np.ndarray)

    # 2. with torch tensors
    source_features_test = torch.rand(n_features_test, n_voxels_source)
    target_features_test = torch.rand(n_features_test, n_voxels_target)
    source_features_on_target = fine_mapping.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, torch.Tensor)
    target_features_on_source = fine_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, torch.Tensor)

    source_features_test = torch.rand(n_voxels_source)
    target_features_test = torch.rand(n_voxels_target)
    source_features_on_target = fine_mapping.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, torch.Tensor)
    target_features_on_source = fine_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, torch.Tensor)
