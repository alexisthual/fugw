# %%
"""
Monitor the Pearson correlation during training with callbacks
==============================================================

In this example, we use a callback function to monitor the Pearson correlation
between transformed and target features at each iteration of
the block-coordinate descent (BCD) algorithm.
This can be useful to detect numerical errors, or to check that the
mapping is not over-fitting training data.
"""
# sphinx_gallery_thumbnail_number = 1

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from nilearn import datasets, image
from rich.console import Console
from scipy.spatial import distance_matrix

from fugw.mappings import FUGW
from fugw.utils import _make_tensor

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


# %%
# We then compute the distance matrix between voxel coordinates.
source_geometry = distance_matrix(coordinates, coordinates)
target_geometry = source_geometry.copy()

# %%
# In order to avoid numerical errors when fitting the mapping, we normalize
# all feature and geometry arrays.
source_features_normalized = source_features / np.linalg.norm(
    source_features, axis=1
).reshape(-1, 1)
target_features_normalized = target_features / np.linalg.norm(
    target_features, axis=1
).reshape(-1, 1)
source_geometry_normalized = source_geometry / np.max(source_geometry)
target_geometry_normalized = target_geometry / np.max(target_geometry)


# %%
# We first define a function to compute the Pearson correlation
# between two tensors. Such function is not available in PyTorch,
# but it is easy to implement.
def pearson_corr(x, y, plan):
    """
    Compute the Pearson correlation between transformed
    source features and target features.
    """
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        x = torch.tensor(x).to(torch.float64)
        y = torch.tensor(y).to(torch.float64)

    # Compute the transformed features
    x_transformed = (
        (plan @ x.T / plan.sum(dim=1).reshape(-1, 1)).T.detach().cpu()
    )

    vx = x_transformed - torch.mean(x_transformed)
    vy = y - torch.mean(y)

    corr = (
        torch.sum(vx * vy)
        / (torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2)))
    ).numpy()

    return corr


# %%
# We then define a callback function that computes the
# Pearson correlation between transformed features and
# target features at each BCD iteration.


# Initialize the transport plan with ones and normalize it
init_plan = torch.ones(
    (source_features_normalized.shape[1], source_features_normalized.shape[1])
).to(float)
init_plan_normalized = init_plan / init_plan.sum()

# Initialize the list of Pearson correlations by fitting
# source features with the initial plan
corr_bcd_steps = [
    pearson_corr(
        source_features_normalized,
        target_features_normalized,
        init_plan_normalized,
    )
]


def correlation_callback(
    locals,
    source_features=None,
    target_features=None,
    device=torch.device("cpu"),
):
    console = Console()

    # Get current transport plan and tensorize features
    pi = locals["pi"]
    source_features_tensor = _make_tensor(source_features, device)
    target_features_tensor = _make_tensor(target_features, device)

    # Compute the Pearson correlation and append it to the list
    corr = pearson_corr(source_features_tensor, target_features_tensor, pi)
    corr_bcd_steps.append(corr)

    console.log("Pearson correlation: ", corr)


# %%
# We now fit the mapping using the sinkhorn solver and 10 BCD iterations.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mapping = FUGW(alpha=0.5, rho=1, eps=1e-4)
_ = mapping.fit(
    source_features=source_features_normalized,
    target_features=target_features_normalized,
    source_geometry=source_geometry_normalized,
    target_geometry=target_geometry_normalized,
    solver="sinkhorn",
    solver_params={
        "nits_bcd": 10,
    },
    callback_bcd=partial(
        correlation_callback,
        source_features=source_features_normalized,
        target_features=target_features_normalized,
        device=device,
    ),
    verbose=True,
)

# %%
# The Pearson correlation and training loss evolution are then plotted for
# each BCD iteration. As we begin over-fitting the training data, the
# correlation rises during the initial iterations before gradually
# falling down.
fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("BCD Step")
ax1.set_ylabel("FUGW loss", color=color)
ax1.plot(mapping.loss_steps, mapping.loss["total"], color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = "tab:blue"
ax2.set_ylabel("Pearson correlation", color=color)
ax2.plot(
    mapping.loss_steps[: len(corr_bcd_steps)], corr_bcd_steps, color=color
)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Evolution of training loss and Pearson correlation")
fig.tight_layout()
plt.show()

# %%
