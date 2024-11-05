import torch
from sklearn.preprocessing import OneHotEncoder


def check_labels(labels: torch.Tensor) -> None:
    """
    Check that labels are a 1D tensor of integers.

    Parameters
    ----------
    labels: torch.Tensor
        Labels to check.

    Raises
    ------
    ValueError
        If labels are not a 1D tensor of integers.
    """
    if not torch.is_tensor(labels):
        raise ValueError(f"labels must be a tensor, got {type(labels)}.")
    if labels.dim() != 1:
        raise ValueError(f"labels must be a 1D tensor, got {labels.dim()}D.")
    if labels.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError(
            f"labels must be an integer tensor, got {labels.dtype}."
        )


def one_hot_encoding(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute one-hot encoding of the labels.

    Parameters
    ----------
    labels: torch.Tensor of size (n,)
        Cluster labels for each voxel.
        Must be a 1D tensor of c integers values.

    Returns
    -------
    one_hot: torch.sparse_coo_tensor of size (n, c).
        One-hot encoding of the labels.
    """
    # Convert labels to string
    labels_categorical = labels.cpu().numpy().astype(str).reshape(-1, 1)
    # Use sklearn to compute the one-hot encoding
    encoder = OneHotEncoder(sparse_output=False)
    one_hot = encoder.fit_transform(labels_categorical)
    one_hot_tensor = torch.from_numpy(one_hot)
    return one_hot_tensor.to_sparse_coo().to(labels.device)


def compute_sparsity_mask(
    labels: torch.Tensor,
    device: str = "auto",
) -> torch.Tensor:
    """
    Compute sparsity mask from coarse mapping.

    Parameters
    ----------
    labels: torch.Tensor of size (n,)
        Cluster labels for each voxel.
    device: "auto" or torch.device
        if "auto": use first available gpu if it's available,
        cpu otherwise.

    Returns
    -------
    sparsity_mask: torch.sparse_coo_tensor of size (n, m)
        Sparsity mask used to initialize the fine mapping.
    """
    check_labels(labels)

    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
    labels = labels.to(device)

    # Create a one-hot encoding of the voxels
    one_hot = one_hot_encoding(labels)
    return (one_hot @ one_hot.T).coalesce().to(torch.float32)
