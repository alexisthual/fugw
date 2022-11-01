import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin


class BaseModel(BaseEstimator, TransformerMixin):
    def fit(self, source_data, target_data):
        return None

    def transform(self, source_map):
        return None


def make_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.tensor(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise Exception(f"Expected np.array or torch.tensor, got {type(x)}")
