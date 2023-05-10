# %%
"""
===================================================================
Align low-resolution brain volumes of 2 individuals with fMRI data
===================================================================

In this example, we align 2 low-resolution brain volumes
using 4 fMRI feature maps (z-score contrast maps).
"""
# sphinx_gallery_thumbnail_number = 7

import numpy as np
import matplotlib.pyplot as plt

from nilearn import datasets, image
from scipy.spatial import distance_matrix
from fugw.mappings import FUGW

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
n_training_contrasts = 3

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
# We then downsample the images by 3 to reduce the computational cost.
SCALE_FACTOR = 4

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
# We then compute the distance matrix between voxel coordinates.
source_geometry = distance_matrix(coordinates, coordinates)
target_geometry = source_geometry.copy()

# %%
# In order to avoid numerical errors when fitting the mapping, we normalize
# both the features and the geometry.
source_features_normalized = source_features / np.linalg.norm(
    source_features, axis=1
).reshape(-1, 1)
target_features_normalized = target_features / np.linalg.norm(
    target_features, axis=1
).reshape(-1, 1)
source_geometry_normalized = source_geometry / np.max(source_geometry)
target_geometry_normalized = target_geometry / np.max(target_geometry)

# %%
# We now fit the mapping using the sinkhorn solver and 3 BCD iterations.
mapping = FUGW(alpha=0.5, rho=1, eps=1e-4)
_ = mapping.fit(
    source_features=source_features_normalized[:n_training_contrasts],
    target_features=target_features_normalized[:n_training_contrasts],
    source_geometry=source_geometry_normalized,
    target_geometry=target_geometry_normalized,
    source_features_val=source_features_normalized[n_training_contrasts:],
    target_features_val=target_features_normalized[n_training_contrasts:],
    source_geometry_val=source_geometry_normalized,
    target_geometry_val=target_geometry_normalized,
    solver="sinkhorn",
    solver_params={
        "nits_bcd": 10,
    },
    verbose=True,
)

# %%
# Plot the evolution of losses on train and test datasets.
fig, ax1 = plt.subplots()

color = "tab:blue"
ax1.set_xlabel("BCD Step")
ax1.set_ylabel("FUGW loss train", color=color)
ax1.plot(mapping.loss_steps, mapping.loss["total"], color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = "tab:orange"
ax2.set_ylabel(
    "FUGW loss test", color=color
)  # we already handled the x-label with ax1
ax2.plot(mapping.loss_steps, mapping.validation_loss["total"], color=color)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Training and validation losses")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# %%
