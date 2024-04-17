import torch

from fugw.mappings.dense import FUGW
from fugw.mappings.sparse import FUGWSparse
from fugw.scripts import coarse_to_fine
from fugw.utils import _make_tensor


class FUGWSparseBarycenter:
    """FUGW sparse barycenters"""

    def __init__(
        self,
        alpha_coarse=0.5,
        alpha_fine=0.5,
        rho_coarse=1,
        rho_fine=1e-2,
        eps_coarse=1e-4,
        eps_fine=1e-4,
        selection_radius=10,
        reg_mode="joint",
        force_psd=False,
        learn_geometry=False,
    ):
        # Save model arguments
        self.alpha_coarse = alpha_coarse
        self.alpha_fine = alpha_fine
        self.rho_coarse = rho_coarse
        self.rho_fine = rho_fine
        self.eps_coarse = eps_coarse
        self.eps_fine = eps_fine
        self.reg_mode = reg_mode
        self.force_psd = force_psd
        self.learn_geometry = learn_geometry
        self.selection_radius = selection_radius

    @staticmethod
    def update_barycenter_features(plans, weights_list, features_list, device):
        barycenter_features = 0
        for i, (pi, weights, features) in enumerate(
            zip(plans, weights_list, features_list)
        ):
            w = _make_tensor(weights, device=device)
            f = _make_tensor(features, device=device)

            if features is not None:
                acc = w * pi.T @ f.T / pi.sum(0).reshape(-1, 1)

                if i == 0:
                    barycenter_features = acc
                else:
                    barycenter_features += acc

        return barycenter_features.T

    @staticmethod
    def get_dim(C):
        if isinstance(C, tuple):
            return C[0].shape[0]
        elif torch.is_tensor(C):
            return C.shape[0]

    @staticmethod
    def get_device_dtype(C):
        if isinstance(C, tuple):
            return C[0].device, C[0].dtype
        elif torch.is_tensor(C):
            return C.device, C.dtype

    def compute_all_ot_plans(
        self,
        plans,
        weights_list,
        features_list,
        geometry_list,
        barycenter_weights,
        barycenter_features,
        barycenter_geometry_embedding,
        mesh_sample,
        solver,
        coarse_mapping_solver_params,
        fine_mapping_solver_params,
        selection_radius,
        callback_bcd,
        device,
        verbose,
    ):
        new_plans = []
        new_losses = []

        for i, (features, weights) in enumerate(
            zip(features_list, weights_list)
        ):
            if len(geometry_list) == 1 and len(weights_list) > 1:
                G = geometry_list[0]
            else:
                G = geometry_list[i]

            coarse_mapping = FUGW(
                alpha=self.alpha_coarse,
                rho=self.rho_coarse,
                eps=self.eps_coarse,
                reg_mode=self.reg_mode,
            )

            fine_mapping = FUGWSparse(
                alpha=self.alpha_fine,
                rho=self.rho_fine,
                eps=self.eps_fine,
                reg_mode=self.reg_mode,
            )

            coarse_to_fine.fit(
                source_features=features,
                target_features=barycenter_features,
                source_geometry_embeddings=G,
                target_geometry_embeddings=barycenter_geometry_embedding,
                source_sample=mesh_sample,
                target_sample=mesh_sample,
                coarse_mapping=coarse_mapping,
                source_weights=weights,
                target_weights=barycenter_weights,
                coarse_mapping_solver=solver,
                coarse_mapping_solver_params=coarse_mapping_solver_params,
                coarse_pairs_selection_method="topk",
                source_selection_radius=selection_radius,
                target_selection_radius=selection_radius,
                fine_mapping=fine_mapping,
                fine_mapping_solver=solver,
                fine_mapping_solver_params=fine_mapping_solver_params,
                init_plan=plans[i] if plans is not None else None,
                device=device,
                verbose=verbose,
            )

            new_plans.append(fine_mapping.pi)
            new_losses.append(
                (
                    fine_mapping.loss,
                    fine_mapping.loss_steps,
                    fine_mapping.loss_times,
                )
            )

        return new_plans, new_losses

    def fit(
        self,
        weights_list,
        features_list,
        geometry_list,
        barycenter_size=None,
        init_barycenter_weights=None,
        init_barycenter_features=None,
        init_barycenter_geometry=None,
        solver="sinkhorn",
        coarse_mapping_solver_params={},
        fine_mapping_solver_params={},
        mesh_sample=None,
        callback_bcd=None,
        nits_barycenter=5,
        device="auto",
        verbose=False,
    ):
        """Compute barycentric features and geometry
        minimizing FUGW loss to list of distributions given as input.
        In this documentation, we refer to a single distribution as
        an a subject's or an individual's distribution.

        Parameters
        ----------
        weights_list (list of np.array): List of weights. Different individuals
            can have weights with different sizes.
        features_list (list of np.array): List of features. Individuals should
            have the same number of features n_features.
        geometry_list (list of np.array or np.array): List of kernel matrices
            or just one kernel matrix if it's shared across individuals
            barycenter_size (int, optional): Size of computed
            barycentric features and geometry. Defaults to None.
        init_barycenter_weights (np.array, optional): Distribution weights
            of barycentric points. If None, points will have uniform
            weights. Defaults to None.
        mesh_sample (np.array, optional): Sample points on which to compute
            the barycenter. Defaults to None.
        init_barycenter_features (np.array, optional): np.array of size
            (barycenter_size, n_features). Defaults to None.
        init_barycenter_geometry (np.array, optional): np.array of size
            (barycenter_size, barycenter_size). Defaults to None.
        device: "auto" or torch.device
            if "auto": use first available gpu if it's available,
            cpu otherwise.
        callback_bcd: callable, optional.
            Callback function to call at each barycenter computation step.
            Defaults to None.

        Returns
        -------
        barycenter_weights: np.array of size (barycenter_size)
        barycenter_features: np.array of size (barycenter_size, n_features)
        barycenter_geometry: np.array of size
            (barycenter_size, barycenter_size)
        plans: list of arrays
        duals: list of (array, array)
        losses_each_bar_step: list such that l[s][i]
            is a tuple containing:
                - loss
                - loss_steps
                - loss_times
            for individual i at barycenter computation step s
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if barycenter_size is None:
            barycenter_size = weights_list[0].shape[0]

        # Initialize barycenter weights, features and geometry
        if init_barycenter_weights is None:
            barycenter_weights = (
                torch.ones(barycenter_size) / barycenter_size
            ).to(device)
        else:
            barycenter_weights = _make_tensor(
                init_barycenter_weights, device=device
            )

        if init_barycenter_features is None:
            barycenter_features = torch.ones(
                (features_list[0].shape[0], barycenter_size)
            ).to(device)
            barycenter_features = barycenter_features / torch.norm(
                barycenter_features, dim=1
            ).reshape(-1, 1)
        else:
            barycenter_features = _make_tensor(
                init_barycenter_features, device=device
            )

        if init_barycenter_geometry is None:
            barycenter_geometry_embedding = geometry_list[0]
        else:
            barycenter_geometry_embedding = _make_tensor(
                init_barycenter_geometry, device=device
            )

        plans = None
        duals = None
        losses_each_bar_step = []

        for _ in range(nits_barycenter):
            # Transport all elements
            plans, losses = self.compute_all_ot_plans(
                plans,
                weights_list,
                features_list,
                geometry_list,
                barycenter_weights,
                barycenter_features,
                barycenter_geometry_embedding,
                mesh_sample,
                solver,
                coarse_mapping_solver_params,
                fine_mapping_solver_params,
                self.selection_radius,
                callback_bcd,
                device,
                verbose,
            )

            losses_each_bar_step.append(losses)

            # Update barycenter features and geometry
            barycenter_features = self.update_barycenter_features(
                plans, weights_list, features_list, device
            )

        return (
            barycenter_weights,
            barycenter_features,
            plans,
            duals,
            losses_each_bar_step,
        )
