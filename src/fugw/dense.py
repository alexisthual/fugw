import numpy as np
import torch

from fugw.solvers.dense import FUGWSolver
from fugw.utils import BaseModel, make_tensor


class FUGW(BaseModel):
    def __init__(
        self,
        alpha=0.5,
        rho=1,
        eps=1e-2,
        uot_solver="sinkhorn",
        reg_mode="joint",
        verbose=False,
        early_stopping_threshold=1e-6,
        **kwargs,
    ):
        # Save model arguments
        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.uot_solver = uot_solver
        self.reg_mode = reg_mode
        self.verbose = verbose
        self.early_stopping_threshold = early_stopping_threshold

    def fit(
        self,
        source_features,
        target_features,
        source_geometry=None,
        target_geometry=None,
        source_weights=None,
        target_weights=None,
        init_plan=None,
        init_duals=None,
        device="auto",
        **kwargs,
    ):
        """
        Compute transport plan between source and target distributions
        using feature maps and geometries.
        In our case, feature maps are fMRI contrast maps,
        and geometries are that of the cortical meshes of individuals
        under study.

        Parameters
        ----------
        source_features: ndarray(n_features, n1)
            Feature maps for source subject.
            n_features is the number of contrast maps, it should
            be the same for source and target data.
            n1 is the number of nodes on the source graph, it
            can be different from n2, the number of nodes on the
            target graph.
        target_features: ndarray(n_features, n2)
            Feature maps for target subject.
        source_geometry: ndarray(n1, n1)
            Kernel matrix of anatomical distances
            between nodes of source mesh
        target_geometry: ndarray(n2, n2)
            Kernel matrix of anatomical distances
            between nodes of target mesh
        source_weights: ndarray(n1) or None
            Distribution weights of source nodes.
            Should sum to 1. If None, eahc node's weight
            will be set to 1 / n1.
        target_weights: ndarray(n1) or None
            Distribution weights of target nodes.
            Should sum to 1. If None, eahc node's weight
            will be set to 1 / n2.
        device: "auto" or torch.device
            if "auto": use first available gpu if it's available,
            cpu otherwise.

        Returns
        -------
        self: FUGW class object
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if isinstance(self.rho, float) or isinstance(self.rho, int):
            rho_x = self.rho
            rho_y = self.rho
        elif isinstance(self.rho, tuple) and len(self.rho) == 2:
            rho_x, rho_y = self.rho
        else:
            raise ValueError(
                "Invalid value of rho. Must be either a scalar or a tuple of"
                " two scalars."
            )

        # Set weights if they were not set by user
        if source_weights is None:
            Ws = (
                torch.ones(source_features.shape[1], device=device)
                / source_features.shape[1]
            )
        else:
            Ws = make_tensor(source_weights, device=device)

        if target_weights is None:
            Wt = (
                torch.ones(target_features.shape[1], device=device)
                / target_features.shape[1]
            )
        else:
            Wt = make_tensor(target_weights, device=device)

        # Compute distance matrix between features
        Fs = make_tensor(source_features.T, device=device)
        Ft = make_tensor(target_features.T, device=device)
        K = torch.cdist(Fs, Ft, p=2) ** 2

        # Load anatomical kernels to GPU
        Gs = make_tensor(source_geometry, device=device)
        Gt = make_tensor(target_geometry, device=device)

        # Create model
        model = FUGWSolver(**kwargs)

        # Compute transport plan
        (
            pi,
            gamma,
            duals_pi,
            duals_gamma,
            loss_steps,
            loss_,
            loss_ent_,
        ) = model.solver(
            px=Ws,
            py=Wt,
            K=K,
            Gs=Gs,
            Gt=Gt,
            alpha=self.alpha,
            rho_x=rho_x,
            rho_y=rho_y,
            eps=self.eps,
            uot_solver=self.uot_solver,
            reg_mode=self.reg_mode,
            early_stopping_threshold=self.early_stopping_threshold,
            init_plan=init_plan,
            init_duals=init_duals,
            verbose=self.verbose,
        )

        # Store variables of interest in model
        self.pi = pi.detach().cpu().numpy()
        self.loss_steps = loss_steps
        self.loss_ = loss_
        self.loss_ent = loss_ent_

        # Free allocated GPU memory
        del Fs, Ft, K, Gs, Gt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (pi, gamma, duals_pi, duals_gamma, loss_steps, loss_, loss_ent_)

    def transform(self, source_features, device="auto"):
        """
        Transport source feature maps using fitted OT plan.
        Use GPUs if available.

        Parameters
        ----------
        source_features: ndarray(n_samples, n1) or ndarray(n1)
            Contrast map for source subject
        device: "auto" or torch.device
            If "auto": use first available GPU if it's available,
            CPU otherwise.

        Returns
        -------
        transported_data: ndarray(n_samples, n2) or ndarray(n2)
            Contrast map transported in target subject's space
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if self.pi is None:
            raise Exception("Model should be fitted before calling transform")

        is_one_dimensional = False
        if source_features.ndim == 1:
            is_one_dimensional = True
            source_features = source_features.reshape(1, -1)
        if source_features.ndim > 2:
            raise ValueError(
                "source_features has too many dimensions:"
                f" {source_features.ndim}"
            )

        # Move data to device if need be
        pi = make_tensor(self.pi, device=device)
        source_features_tensor = make_tensor(source_features, device=device)

        # Transform data
        transformed_data = (
            (pi.T @ source_features_tensor.T / pi.sum(dim=0).reshape(-1, 1))
            .T.detach()
            .cpu()
        )

        # Free allocated GPU memory
        del pi, source_features_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Modify returned tensor so that it matches
        # source_features's shape and python type
        if isinstance(source_features, np.ndarray):
            transformed_data = transformed_data.numpy()

        if transformed_data.ndim > 1 and is_one_dimensional:
            transformed_data = transformed_data.flatten()

        return transformed_data

    def inverse_transform(self, target_features, device="auto"):
        """
        Transport target feature maps using fitted OT plan.
        Use GPUs if available.

        Parameters
        ----------
        target_features: ndarray(n_samples, n2) or ndarray(n2)
            Contrast map for target subject
        device: "auto" or torch.device
            If "auto": use first available GPU if it's available,
            CPU otherwise.

        Returns
        -------
        transported_data: ndarray(n_samples, n1) or ndarray(n1)
            Contrast map transported in source subject's space
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if self.pi is None:
            raise Exception("Model should be fitted before calling transform")

        is_one_dimensional = False
        if target_features.ndim == 1:
            is_one_dimensional = True
            target_features = target_features.reshape(1, -1)
        if target_features.ndim > 2:
            raise ValueError(
                "target_features has too many dimensions:"
                f" {target_features.ndim}"
            )

        # Move data to device if need be
        pi = make_tensor(self.pi, device=device)
        target_features_tensor = make_tensor(target_features, device=device)

        # Transform data
        transformed_data = (
            (pi @ target_features_tensor.T / pi.sum(dim=1).reshape(-1, 1))
            .T.detach()
            .cpu()
        )

        # Free allocated GPU memory
        del pi, target_features_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Modify returned tensor so that it matches
        # target_features's  shape and python type
        if isinstance(target_features, np.ndarray):
            transformed_data = transformed_data.numpy()

        if transformed_data.ndim > 1 and is_one_dimensional:
            transformed_data = transformed_data.flatten()

        return transformed_data
