"""
Transport distributions containing more than 10k points
=======================================================
"""

##############################################################################
import torch

from fugw.scripts import coarse_to_fine
from fugw.mappings import FUGW, FUGWSparse
from fugw.mappings.utils import init_mock_distribution

torch.manual_seed(0)

n_voxels_source = 300
n_samples_source = 150
n_voxels_target = 300
n_samples_target = 150
n_features_train = 10
n_features_test = 5

# Generate random training data for source and target distributions
_, source_features, _, source_embeddings = init_mock_distribution(
    n_features_train, n_voxels_source
)
_, target_features, _, target_embeddings = init_mock_distribution(
    n_features_train, n_voxels_target
)

# Define the optimization problem to solve
coarse_model = FUGW(alpha=0.5)
fine_model = FUGWSparse(alpha=0.5)

# Specify which solvers to use at each step
coarse_model_fit_params = {
    "uot_solver": "mm",
    "tol_uot": 1e-10,
    "nits_uot": 100,
}

fine_model_fit_params = {
    "uot_solver": "mm",
    "tol_uot": 1e-10,
}

# Fit transport plans
coarse_to_fine.fit(
    coarse_model=coarse_model,
    coarse_model_fit_params=coarse_model_fit_params,
    coarse_pairs_selection_method="topk",
    source_selection_radius=1,
    target_selection_radius=1,
    fine_model=fine_model,
    fine_model_fit_params=fine_model_fit_params,
    source_sample_size=n_samples_source,
    target_sample_size=n_samples_target,
    source_features=source_features,
    target_features=target_features,
    source_geometry_embeddings=source_embeddings,
    target_geometry_embeddings=target_embeddings,
    verbose=True,
)

# Both the coarse and fine-scale transport plans can be accessed
# after the models have been fitted
print(f"Coarse transport plan's total mass: {coarse_model.pi.sum()}")
print(
    "Fine-scale transport plan's total mass:"
    f" {torch.sparse.sum(fine_model.pi)}"
)

# Finally, the fitted fine model can transport unseen data
# between source and target
source_features_test = torch.rand(n_features_test, n_voxels_source)
target_features_test = torch.rand(n_features_test, n_voxels_target)
transformed_data = fine_model.transform(source_features_test)
assert transformed_data.shape == target_features_test.shape
