# %%
"""
===================================================================
Align low-resolution brain volumes of 2 individuals with fMRI data
===================================================================

In this example, we align 2 low-resolution brain volumes
using 4 fMRI feature maps (z-score contrast maps).
"""
# sphinx_gallery_thumbnail_number = 8

import numpy as np
import matplotlib.pyplot as plt

from nilearn import datasets, image
from fugw.mappings import FUGW, FUGWSparse
from fugw.scripts import coarse_to_fine, lmds

plt.rcParams["figure.dpi"] = 300

# %%
# We first fetch 5 contrasts for each subject from the localizer dataset.
n_subjects = 2

contrasts = [
    "sentence reading vs checkerboard",
    "sentence listening",
    "calculation vs sentences",
    "left vs right button press",
    "checkerboard",
]
n_training_contrasts = 4

brain_data = datasets.fetch_localizer_contrasts(
    contrasts,
    n_subjects=n_subjects,
    get_anats=True,
)

source_imgs_paths = brain_data["cmaps"][0 : len(contrasts)]
target_imgs_paths = brain_data["cmaps"][len(contrasts) : 2 * len(contrasts)]

source_im = image.load_img(source_imgs_paths)
target_im = image.load_img(target_imgs_paths)

# %%
# Let's use a geometry of 7633 voxels.
SCALE_FACTOR = 2

source_maps = np.nan_to_num(
    source_im.get_fdata()[::SCALE_FACTOR, ::SCALE_FACTOR, ::SCALE_FACTOR]
)
target_maps = np.nan_to_num(
    target_im.get_fdata()[::SCALE_FACTOR, ::SCALE_FACTOR, ::SCALE_FACTOR]
)

segmentation_fine = np.logical_not(np.isnan(source_im.get_fdata()[:, :, :, 0]))
segmentation_coarse = segmentation_fine[
    ::SCALE_FACTOR, ::SCALE_FACTOR, ::SCALE_FACTOR
]
coordinates = np.array(np.nonzero(segmentation_coarse)).T

source_features = source_maps[
    coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
].T
target_features = target_maps[
    coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
].T

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], marker="o")
ax.view_init(10, 135)
ax.set_axis_off()
plt.show()

# %%
# We then compute the distance embedding between the coordinates of the voxels.
source_geometry_embeddings = lmds.compute_lmds_volume(
    segmentation_coarse
).nan_to_num()
target_geometry_embeddings = source_geometry_embeddings.clone()

# Show the embedding shape
print(source_geometry_embeddings.shape)

# %%
# In order to avoid numerical errors when fitting the mapping, we normalize
# both the features and the geometry.
source_features_normalized = source_features / np.linalg.norm(
    source_features, axis=1
).reshape(-1, 1)
target_features_normalized = target_features / np.linalg.norm(
    target_features, axis=1
).reshape(-1, 1)

source_embeddings_normalized, source_distance_max = (
    coarse_to_fine.random_normalizing(source_geometry_embeddings)
)
target_embeddings_normalized, target_distance_max = (
    coarse_to_fine.random_normalizing(target_geometry_embeddings)
)

# %%
# We now fit the mapping using the sinkhorn solver and 3 BCD iterations.
alpha_coarse = 0.5
rho_coarse = 1
eps_coarse = 1e-4
coarse_mapping = FUGW(alpha=alpha_coarse, rho=rho_coarse, eps=eps_coarse)
coarse_mapping_solver = "mm"
coarse_mapping_solver_params = {
    "nits_bcd": 5,
    "tol_uot": 1e-10,
}

alpha_fine = 0.5
rho_fine = 1
eps_fine = 1e-4
fine_mapping = FUGWSparse(alpha=alpha_fine, rho=rho_fine, eps=eps_fine)
fine_mapping_solver = "mm"
fine_mapping_solver_params = {
    "nits_bcd": 3,
    "tol_uot": 1e-10,
}

# %%
# Let's subsample the vertices.
source_sample = coarse_to_fine.sample_volume_uniformly(
    segmentation_coarse,
    embeddings=source_geometry_embeddings,
    n_samples=1000,
)
target_sample = coarse_to_fine.sample_volume_uniformly(
    segmentation_coarse,
    embeddings=target_geometry_embeddings,
    n_samples=1000,
)

# %%
# Train both the coarse and the fine mapping.
coarse_to_fine.fit(
    # Source and target's features and embeddings
    source_features=source_features_normalized[:n_training_contrasts, :],
    target_features=target_features_normalized[:n_training_contrasts, :],
    source_geometry_embeddings=source_embeddings_normalized,
    target_geometry_embeddings=target_embeddings_normalized,
    # Parametrize step 1 (coarse alignment between source and target)
    source_sample=source_sample,
    target_sample=target_sample,
    coarse_mapping=coarse_mapping,
    coarse_mapping_solver=coarse_mapping_solver,
    coarse_mapping_solver_params=coarse_mapping_solver_params,
    # Parametrize step 2 (selection of pairs of indices present in
    # fine-grained's sparsity mask)
    coarse_pairs_selection_method="topk",
    source_selection_radius=10 / source_distance_max,
    target_selection_radius=10 / target_distance_max,
    # Parametrize step 3 (fine-grained alignment)
    fine_mapping=fine_mapping,
    fine_mapping_solver=fine_mapping_solver,
    fine_mapping_solver_params=fine_mapping_solver_params,
    # Misc
    verbose=True,
)

# %%
# Let's plot the probability map of the target voxels being matched with
# the 300th source voxel.
pi = fine_mapping.pi
vertex_index = 300
one_hot = np.zeros(source_features.shape[1])
one_hot[vertex_index] = 1.0
probability_map = fine_mapping.inverse_transform(one_hot)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    coordinates[:, 2],
    marker="o",
    c=probability_map,
    cmap="twilight",
    alpha=0.75,
)
ax.text(
    coordinates[vertex_index, 0],
    coordinates[vertex_index, 1],
    coordinates[vertex_index, 2] - 2,
    "x Source point",
    color="red",
)
ax.view_init(10, 135, 2)
ax.set_title(
    "Probability map of target vertices\n"
    f"being matched with source point {vertex_index}"
)
ax.set_axis_off()
fig.colorbar(plt.cm.ScalarMappable(cmap="twilight"), ax=ax)
plt.show()

# %%
# We can now align the test contrasts using the fitted fine mapping.
contrast_index = -1
predicted_target_features = fine_mapping.transform(
    source_features[contrast_index, :]
)
predicted_target_features.shape

# %%
# Let's compare the Pearson correlation between the source and target features.
corr_pre_mapping = np.corrcoef(
    source_features[contrast_index, :], target_features[contrast_index, :]
)[0, 1]
corr_post_mapping = np.corrcoef(
    predicted_target_features, target_features[contrast_index, :]
)[0, 1]
print(f"Pearson Correlation pre-mapping: {corr_pre_mapping:.2f}")
print(f"Pearson Correlation post-mapping: {corr_post_mapping:.2f}")
print(
    "Relative improvement:"
    f" {(corr_post_mapping - corr_pre_mapping) / corr_pre_mapping *100 :.2f} %"
)


# %%
# Let's plot the transporting feature maps of the test set.
fig = plt.figure(figsize=plt.figaspect(0.3))
fig.suptitle("Transporting feature maps of the test set", size=16)
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    coordinates[:, 2],
    marker="o",
    c=source_features_normalized[-1, :],
    cmap="twilight",
)
ax.view_init(10, 135, 2)
ax.set_title("Source features")
ax.set_axis_off()

ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    coordinates[:, 2],
    marker="o",
    c=predicted_target_features,
    cmap="twilight",
)
ax.view_init(10, 135, 2)
ax.set_title("Predicted target features")
ax.set_axis_off()

ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    coordinates[:, 2],
    marker="o",
    c=target_features_normalized[-1, :],
    cmap="twilight",
)
ax.view_init(10, 135, 2)
ax.set_title("Actual target features")
ax.set_axis_off()
fig.colorbar(plt.cm.ScalarMappable(cmap="twilight"), ax=ax)
plt.show()
