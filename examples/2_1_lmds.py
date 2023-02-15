import meshzoo
import torch

from fugw.scripts import lmds

torch.manual_seed(0)

coordinates, triangles = meshzoo.octa_sphere(30)
n_landmarks = 100
k = 3

X = lmds.compute_lmds(
    coordinates,
    triangles,
    n_landmarks=n_landmarks,
    k=k,
    n_jobs=2,
    verbose=True,
)

assert X.shape == (coordinates.shape[0], k)
