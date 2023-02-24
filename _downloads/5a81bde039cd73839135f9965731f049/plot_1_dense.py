"""
Transport distributions containing less than 10k points
=======================================================
"""

##############################################################################
import torch

from fugw.mappings import FUGW
from fugw.mappings.utils import init_mock_distribution

torch.manual_seed(0)

##############################################################################
n_vertices_source = 1000
n_vertices_target = 900
n_features_train = 10
n_features_test = 5

##############################################################################
# Generate random training data for source and target distributions
_, source_features_train, source_geometry, _ = init_mock_distribution(
    n_features_train, n_vertices_source
)
_, target_features_train, target_geometry, _ = init_mock_distribution(
    n_features_train, n_vertices_target
)

##############################################################################
# Define the optimization problem to solve
fugw = FUGW(alpha=0.5)

##############################################################################
# Fit transport plan between source and target distributions
# with sinkhorn solver
_ = fugw.fit(
    source_features_train,
    target_features_train,
    source_geometry=source_geometry,
    target_geometry=target_geometry,
    uot_solver="sinkhorn",
)

##############################################################################
# The transport plan can be accessed after the model has been fitted
print(f"Transport plan's total mass: {fugw.pi.sum()}")

##############################################################################
# Finally, the fitted model can transport unseen data
# between source and target
source_features_test = torch.rand(n_features_test, n_vertices_source)
target_features_test = torch.rand(n_features_test, n_vertices_target)
transformed_data = fugw.transform(source_features_test)
assert transformed_data.shape == target_features_test.shape
