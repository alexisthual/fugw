import numpy as np
import torch
from scipy.stats import pearsonr

from fugw.solvers.sparse import FUGWSparseSolver
from fugw.utils import (
    BaseModel,
    low_rank_squared_l2,
    make_sparse_tensor,
    make_tensor,
)


class FUGWSparse(BaseModel):
    def __init__(
        self,
        alpha=0.5,
        rho=1,
        eps=1e-2,
        uot_solver="dc",
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
        source_geometry_embedding=None,
        target_geometry_embedding=None,
        source_weights=None,
        target_weights=None,
        init_plan=None,
        init_duals=None,
        return_plans_only=True,
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
        source_geometry_embedding: ndarray(n1, d), optional
            Embedding X such that norm(X_i - X_j) approximates
            the anatomical distance between vertices i and j
            of the source mesh
        target_geometry_embedding: ndarray(n2, d), optional
            Embedding X such that norm(X_i - X_j) approximates
            the anatomical distance between vertices i and j
            of the target mesh
        source_weights: ndarray(n1) or None
            Distribution weights of source nodes.
            Should sum to 1. If None, eahc node's weight
            will be set to 1 / n1.
        target_weights: ndarray(n1) or None
            Distribution weights of target nodes.
            Should sum to 1. If None, eahc node's weight
            will be set to 1 / n2.

        Returns
        -------
        self: FUGW class object
        """
        # Check cuda availability
        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

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
                torch.ones(source_features.shape[1]).type(dtype)
                / source_features.shape[1]
            )
        else:
            Ws = make_tensor(source_weights).type(dtype)

        if target_weights is None:
            Wt = (
                torch.ones(target_features.shape[1]).type(dtype)
                / target_features.shape[1]
            )
        else:
            Wt = make_tensor(target_weights).type(dtype)

        # Compute distance matrix between features
        Fs = make_tensor(source_features.T).type(dtype)
        Ft = make_tensor(target_features.T).type(dtype)
        # K = torch.cdist(Fs, Ft, p=2)
        K1, K2 = low_rank_squared_l2(Fs, Ft)
        K1 = make_tensor(K1).type(dtype)
        K2 = make_tensor(K2).type(dtype)

        # Load anatomical kernels to GPU
        # and normalize them
        Gs1, Gs2 = low_rank_squared_l2(
            source_geometry_embedding, source_geometry_embedding
        )
        Gs1 = make_tensor(Gs1).type(dtype)
        Gs2 = make_tensor(Gs2).type(dtype)
        Gt1, Gt2 = low_rank_squared_l2(
            target_geometry_embedding, target_geometry_embedding
        )
        Gt1 = make_tensor(Gt1).type(dtype)
        Gt2 = make_tensor(Gt2).type(dtype)
        # Gs = make_tensor(source_geometry).type(dtype)
        # Gt = make_tensor(target_geometry).type(dtype)

        # Create model
        model = FUGWSparseSolver(**kwargs)

        # check that all init_plan is valid
        init_plan = make_sparse_tensor(init_plan, dtype)

        # Compute transport plan
        res = model.solver(
            px=Ws,
            py=Wt,
            K=(K1, K2),
            Gs=(Gs1, Gs2),
            Gt=(Gt1, Gt2),
            alpha=self.alpha,
            rho_x=rho_x,
            rho_y=rho_y,
            eps=self.eps,
            uot_solver=self.uot_solver,
            reg_mode=self.reg_mode,
            early_stopping_threshold=self.early_stopping_threshold,
            init_plan=init_plan,
            init_duals=init_duals,
            return_plans_only=return_plans_only,
            verbose=self.verbose,
        )

        self.pi = res[0]

        # Free allocated GPU memory
        del Fs, Ft, K1, K2, Gs1, Gs2, Gt1, Gt2
        if use_cuda:
            torch.cuda.empty_cache()

        return res

    def transform(self, source_features):
        """
        Transport source contrast map using fitted OT plan.
        Use GPUs if available.

        Parameters
        ----------
        source_features: ndarray(n_samples, n1) or ndarray(n1)
            Contrast map for source subject

        Returns
        -------
        transported_data: ndarray(n_samples, n2) or ndarray(n2)
            Contrast map transported in target subject's space
        """
        # Check cuda availability
        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        if self.pi is None:
            raise ("Model should be fitted before calling transform")

        is_one_dimensional = False
        if source_features.ndim == 1:
            is_one_dimensional = True
            source_features = source_features.reshape(1, -1)
        if source_features.ndim > 2:
            raise ValueError(
                "source_features has too many dimensions:"
                f" {source_features.ndim}"
            )

        # Move data to GPU
        source_features_torch = torch.from_numpy(source_features).type(dtype)

        # Transform data
        transformed_data_torch = (
            torch.sparse.mm(
                self.pi.transpose(0, 1), source_features_torch.T
            ).to_dense()
            / torch.sparse.sum(self.pi, dim=0).to_dense().reshape(-1, 1)
        ).T

        # Move transformed data back to CPU
        transformed_data = transformed_data_torch.detach().cpu().numpy()

        # Free allocated GPU memory
        del source_features_torch
        if use_cuda:
            torch.cuda.empty_cache()

        if transformed_data.ndim > 1 and is_one_dimensional:
            transformed_data = transformed_data.flatten()

        return transformed_data

    def score(self, source_features, target_features):
        """
        Transport source contrast maps using fitted OT plan
        and compute correlation with actual target contrast maps.

        Parameters
        ----------
        source_features: ndarray(n_samples, n1)
            Contrast maps for source subject
        target_features: ndarray(n_samples, n2)
            Contrast maps for target subject

        Returns
        -------
        score: float
            Mean correlation across features of self.transform(source_features)
            and target_features
        """

        transported_data = self.transform(source_features)

        score = np.mean(
            [
                pearsonr(transported_data[i, :], target_features[i, :])[0]
                for i in range(transported_data.shape[0])
            ]
        )

        return score
