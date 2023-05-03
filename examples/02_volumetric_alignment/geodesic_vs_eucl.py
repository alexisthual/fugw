# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

from nilearn import datasets, image
from fugw.scripts import lmds


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
assert source_im.shape == target_im.shape


source_features = np.nan_to_num(source_im.get_fdata())
target_features = np.nan_to_num(target_im.get_fdata())


SCALE_FACTOR = 3


source_features = source_features[
    ::SCALE_FACTOR, ::SCALE_FACTOR, ::SCALE_FACTOR
]
target_features = target_features[
    ::SCALE_FACTOR, ::SCALE_FACTOR, ::SCALE_FACTOR
]
data = source_features[:, :, :, 0] != 0


source_features = torch.Tensor(source_features[data].T)
target_features = torch.Tensor(target_features[data].T)


mat = np.array(np.nonzero(data)).T

print(data.shape)

source_embeddings = lmds.compute_lmds(
    torch.Tensor(mat), method="geodesic", n_landmarks=100, k=3
)
target_embeddings = source_embeddings.clone()

plt.imshow(source_embeddings @ source_embeddings.T)
plt.show()
