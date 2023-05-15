# %%
"""
==================================================================
Using a validation set to monitor the convergence of the fugw loss
==================================================================

In this example, we use 3 fMRI feature maps for training and 2 independant
fMRI feature maps for testing to examine the evolutions of a training and
a validation loss on 2 low-resolution brain volumes.
"""
# sphinx_gallery_thumbnail_number = 9

import numpy as np
import matplotlib.pyplot as plt

from nilearn import datasets, image, plotting
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
    get_masks=True,
)

source_imgs_paths = brain_data["cmaps"][0 : len(contrasts)]
target_imgs_paths = brain_data["cmaps"][len(contrasts) : 2 * len(contrasts)]
source_mask = brain_data["masks"][0]

source_im = image.load_img(source_imgs_paths)
target_im = image.load_img(target_imgs_paths)
mask = image.load_img(source_mask)

# %%
# We then downsample the images by 5 to reduce the computational cost.
SCALE_FACTOR = 5

resized_source_affine = source_im.affine.copy() * SCALE_FACTOR
resized_target_affine = target_im.affine.copy() * SCALE_FACTOR

source_im_resized = image.resample_img(source_im, resized_source_affine)
target_im_resized = image.resample_img(target_im, resized_target_affine)
mask_resized = image.resample_img(mask, resized_source_affine)

source_maps = np.nan_to_num(source_im_resized.get_fdata())
target_maps = np.nan_to_num(target_im_resized.get_fdata())
segmentation = mask_resized.get_fdata()

coordinates = np.argwhere(segmentation > 0)

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
ax2.plot(mapping.loss_steps, mapping.loss_val["total"], color=color)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Training and validation losses")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# %%
# Plot the transformation of the first feature map.
example_array = np.nan_to_num(source_im_resized.slicer[..., 0].get_fdata())
example_array /= np.max(np.abs(example_array))
example = image.new_img_like(source_im_resized, example_array)
plotting.view_img_on_surf(example, threshold="90%", surf_mesh="fsaverage5")
# %%
