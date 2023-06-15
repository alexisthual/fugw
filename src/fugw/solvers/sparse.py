from functools import partial
from copy import deepcopy

import time

import numpy as np
import torch

from fugw.solvers.utils import (
    BaseSolver,
    batch_elementwise_prod_and_sum,
    compute_unnormalized_kl,
    compute_unnormalized_kl_sparse,
    compute_divergence_sparse,
    compute_quad_divergence,
    compute_quad_divergence_sparse,
    crow_indices_to_row_indices,
    csr_sum,
    elementwise_prod_fact_sparse,
    solver_sinkhorn_sparse,
    solver_ibpp_sparse,
    solver_mm_sparse,
    solver_mm_l2_sparse,
)
from fugw.utils import _add_dict, console, _make_csr_matrix


class FUGWSparseSolver(BaseSolver):
    """Solver computing sparse solutions"""

    def local_biconvex_cost(
        self, pi, transpose, data_const, tuple_weights, hyperparams
    ):
        """
        Before each block coordinate descent (BCD) step,
        the local cost matrix is updated.
        This local cost is a matrix of size (n, m)
        which evaluates the cost between every pair of points
        of the source and target distributions.
        Then, we run a BCD (sinkhorn, ibpp or mm) step
        which makes use of this cost to update the transport plans.
        """

        rho_s, rho_t, eps, alpha, reg_mode, divergence = hyperparams
        ws, wt, ws_dot_wt = tuple_weights
        device = ws.device

        (
            (Ds_sqr_1, Ds_sqr_2),
            (Dt_sqr_1, Dt_sqr_2),
            (Ds1, Ds2),
            (Dt1, Dt2),
            (K1, K2),
        ) = data_const
        if transpose:
            Ds_sqr_1, Ds_sqr_2 = Ds_sqr_2, Ds_sqr_1
            Dt_sqr_1, Dt_sqr_2 = Dt_sqr_2, Dt_sqr_1
            Ds1, Ds2 = Ds2, Ds1
            Dt1, Dt2 = Dt2, Dt1

        pi1, pi2 = (
            csr_sum(pi, dim=1),
            csr_sum(pi, dim=0),
        )

        # Avoid unnecessary calculation of UGW when alpha = 0
        # row_indices, col_indices = pi._indices()
        crow_indices, col_indices = (
            pi.crow_indices(),
            pi.col_indices(),
        )
        row_indices = crow_indices_to_row_indices(crow_indices)
        cost_values = torch.zeros_like(pi.values()).to(device)

        if alpha != 1 and K1 is not None and K2 is not None:
            wasserstein_cost_values = (
                batch_elementwise_prod_and_sum(
                    K1, K2, row_indices, col_indices, 1
                )
                / 2
            )
            cost_values += (1 - alpha) * wasserstein_cost_values

        # or UOT when alpha = 1
        if alpha != 0:
            A = Ds_sqr_1 @ (Ds_sqr_2.T @ pi1)
            B = Dt_sqr_1 @ (Dt_sqr_2.T @ pi2)
            C1, C2 = Ds1, ((Ds2.T @ torch.sparse.mm(pi, Dt2)) @ Dt1.T).T

            gromov_wasserstein_cost_values = (
                A[row_indices]
                + B[col_indices]
                - 2
                * batch_elementwise_prod_and_sum(
                    C1, C2, row_indices, col_indices, 1
                )
            )

            cost_values += alpha * gromov_wasserstein_cost_values

        if divergence == "kl":
            # or when cost is balanced
            if rho_s != float("inf") and rho_s != 0:
                marginal_cost_dim1 = compute_unnormalized_kl(pi1, ws)
                cost_values += rho_s * marginal_cost_dim1
            if rho_t != float("inf") and rho_t != 0:
                marginal_cost_dim2 = compute_unnormalized_kl(pi2, wt)
                cost_values += rho_t * marginal_cost_dim2

            if reg_mode == "joint":
                cost_values += eps * compute_unnormalized_kl_sparse(
                    pi, ws_dot_wt
                )
        elif divergence == "l2":
            # Marginal constraints do not appear in the cost matrix
            # in the L2 case. See calculations.
            pass

        cost = torch.sparse_csr_tensor(
            crow_indices, col_indices, cost_values, size=pi.size()
        )

        return cost

    def fugw_loss(self, pi, gamma, data_const, tuple_weights, hyperparams):
        """Compute FUGW loss and each of its components.

        Computes scalar fugw loss, which is a combination of:
        - a Wasserstein loss on features
        - a Gromow-Wasserstein loss on geometries
        - marginal constraints on the computed OT plan
        - a regularization term (KL, ie entropic, or L2)

        Parameters
        ----------
        pi: torch.Tensor
        gamma: torch.Tensor
        data_const: tuple
        tuple_weights: tuple
        hyperparams: tuple

        Returns
        -------
        l: dict
            Dictionary containing the loss and its unweighted components.
            Keys are: "wasserstein", "gromov_wasserstein",
            "marginal_constraint_dim1", "marginal_constraint_dim2",
            "regularization", "total".
            Values are float or None.
        """
        rho_s, rho_t, eps, alpha, reg_mode, divergence = hyperparams
        ws, wt, ws_dot_wt = tuple_weights
        (
            (Ds_sqr_1, Ds_sqr_2),
            (Dt_sqr_1, Dt_sqr_2),
            (Ds1, Ds2),
            (Dt1, Dt2),
            (K1, K2),
        ) = data_const

        pi1, pi2 = (
            csr_sum(pi, dim=1),
            csr_sum(pi, dim=0),
        )
        gamma1, gamma2 = (
            csr_sum(gamma, dim=1),
            csr_sum(gamma, dim=0),
        )

        loss_wasserstein = None
        loss_gromov_wasserstein = None
        loss_marginal_constraint_dim1 = None
        loss_marginal_constraint_dim2 = None
        loss_regularization = None
        loss = 0

        if alpha != 1 and K1 is not None and K2 is not None:
            # TODO: warning, torch sparse transforms LongTensor
            # into IntTensor for crow and col indices
            # when adding 2 CSR matrices together.
            # This could become problematic for sparse matrices
            # with more non-null elements than an int can store.
            loss_wasserstein = (
                csr_sum(elementwise_prod_fact_sparse(K1, K2, pi + gamma)) / 2
            )
            loss += (1 - alpha) * loss_wasserstein

        if alpha != 0:
            A = (Ds_sqr_1 @ (Ds_sqr_2.T @ gamma1)).dot(pi1)
            B = (Dt_sqr_1 @ (Dt_sqr_2.T @ gamma2)).dot(pi2)
            C = elementwise_prod_fact_sparse(
                Ds1, ((Ds2.T @ torch.sparse.mm(gamma, Dt2)) @ Dt1.T).T, pi
            )
            loss_gromov_wasserstein = A + B - 2 * csr_sum(C)
            loss += alpha * loss_gromov_wasserstein

        if rho_s != float("inf") and rho_s != 0:
            loss_marginal_constraint_dim1 = compute_quad_divergence(
                pi1, gamma1, ws, ws, divergence
            )
            loss += rho_s * loss_marginal_constraint_dim1
        if rho_t != float("inf") and rho_t != 0:
            loss_marginal_constraint_dim2 = compute_quad_divergence(
                pi2, gamma2, wt, wt, divergence
            )
            loss += rho_t * loss_marginal_constraint_dim2

        if eps != 0:
            if reg_mode == "joint":
                loss_regularization = compute_quad_divergence_sparse(
                    pi, gamma, ws_dot_wt, ws_dot_wt, divergence
                )
            elif reg_mode == "independent":
                loss_regularization = compute_divergence_sparse(
                    pi, ws_dot_wt, divergence
                ) + compute_divergence_sparse(gamma, ws_dot_wt, divergence)

            loss = loss + eps * loss_regularization

        return {
            "wasserstein": loss_wasserstein.item(),
            "gromov_wasserstein": loss_gromov_wasserstein.item(),
            "marginal_constraint_dim1": loss_marginal_constraint_dim1.item(),
            "marginal_constraint_dim2": loss_marginal_constraint_dim2.item(),
            "regularization": loss_regularization.item(),
            "total": loss.item(),
        }

    def get_parameters_uot_l2(self, pi, tuple_weights, hyperparams):
        """Compute parameters of the L2 loss."""
        rho_s, rho_t, eps, reg_mode = hyperparams
        ws, wt, ws_dot_wt = tuple_weights

        pi1, pi2 = csr_sum(pi, dim=1), csr_sum(pi, dim=0)
        l2_pi1, l2_pi2 = (pi1**2).sum(), (pi2**2).sum()
        l2_pi = (pi.values() ** 2).sum()

        # Distribution weights of the updated problem
        weight_ws = pi1.dot(ws) / l2_pi1
        weight_wt = pi2.dot(wt) / l2_pi2
        weight_wst_values = (
            (pi.values() * ws_dot_wt.values()).sum() / l2_pi
            if reg_mode == "joint"
            else 1
        )

        weighted_tuple_weights = (
            weight_ws * ws,
            weight_wt * wt,
            _make_csr_matrix(
                pi.crow_indices(),
                pi.col_indices(),
                weight_wst_values * ws_dot_wt.values(),
                pi.size(),
                pi.device,
            ),
        )

        # Loss parameters of the updated problem
        new_rho_s = rho_s * l2_pi1
        new_rho_t = rho_t * l2_pi2
        new_eps = eps * l2_pi if reg_mode == "joint" else eps
        uot_params = (new_rho_s, new_rho_t, new_eps)

        return weighted_tuple_weights, uot_params

    def solve(
        self,
        alpha=0.5,
        rho_s=1,
        rho_t=1,
        eps=1e-2,
        reg_mode="joint",
        divergence="kl",
        F=(None, None),
        Ds=(None, None),
        Dt=(None, None),
        F_val=(None, None),
        Ds_val=(None, None),
        Dt_val=(None, None),
        ws=None,
        wt=None,
        init_plan=None,
        init_duals=None,
        solver="ibpp",
        callback_bcd=None,
        verbose=False,
    ):
        """Run BCD iterations.

        Parameters
        ----------
        alpha: float, optional
        rho_s: float, optional
        rho_t: float, optional
        eps: float, optional
        reg_mode: string, optional
        divergence: string, optional
        F: (ndarray(n, d+2), ndarray(m, d+2)) or (None, None)
        Ds: (ndarray(n, k+2), ndarray(n, k+2)), or (None, None)
        Dt: (ndarray(m, k+2), ndarray(m, k+2)), or (None, None)
        F_val: (ndarray(n, d+2), ndarray(m, d+2)) or (None, None)
        Ds_val: (ndarray(n, k+2), ndarray(n, k+2)), or (None, None)
        Dt_val: (ndarray(m, k+2), ndarray(m, k+2)), or (None, None)
        ws: ndarray(n), None
            Measures assigned to source points.
        wt: ndarray(m), None
            Measures assigned to target points.
        init_plan: torch.tensor sparse, None
            Initialisation matrix for sample coupling.
        init_duals: torch.tensor sparse, None
            Initialisation matrix for sample coupling.
        solver: "sinkhorn", "mm", "ibpp"
            Solver to use.
        callback_bcd: callable or None
            Callback function called at the end of each BCD step.
            It will be called with the following arguments:

                - locals (dictionary containing all local variables)
        verbose: bool, optional, defaults to False
            Log solving process.

        Returns
        -------
        res: dict
            Dictionary containing the following keys:
                pi: sparse torch.Tensor of size n x m
                    Sample matrix.
                gamma: sparse torch.Tensor of size d1 x d2
                    Feature matrix.
                duals_pi: tuple of torch.Tensor of size
                    Duals of pi
                duals_gamma: tuple of torch.Tensor of size
                    Duals of gamma
                loss: dict of lists
                    Dictionary containing the loss and its unweighted
                    components for each step of the block-coordinate-descent
                    for which the FUGW loss was evaluated.
                    Keys are: "wasserstein", "gromov_wasserstein",
                    "marginal_constraint_dim1", "marginal_constraint_dim2",
                    "regularization", "total".
                    Values are float or None.
                loss_steps: list
                    BCD steps at the end of which the FUGW loss was evaluated
                loss_times: list
                    Elapsed time at the end of each BCD step for which the
                    FUGW loss was evaluated.
        """
        if rho_s == float("inf") and rho_t == float("inf") and eps == 0:
            raise ValueError(
                "This package does not handle balanced cases "
                "(ie infinite values of rho). "
                "You should have a look at POT (https://pythonot.github.io) "
                "and in particular at ot.gromov.fused_gromov_wasserstein "
                "(https://pythonot.github.io/gen_modules/ot.gromov.html"
                "#ot.gromov.fused_gromov_wasserstein)"
            )
        elif eps == 0 and (
            (rho_s == 0 and rho_t == float("inf"))
            or (rho_s == 0 and rho_t == float("inf"))
        ):
            raise ValueError(
                "Invalid rho and eps. Unregularized semi-relaxed GW is not "
                "supported."
            )

        # sanity check
        if solver == "mm" and (rho_s == float("inf") or rho_t == float("inf")):
            solver = "ibpp"
        if solver == "sinkhorn" and eps == 0:
            solver = "ibpp"

        n, m = Ds[0].shape[0], Dt[0].shape[0]
        device, dtype = Ds[0].device, Ds[0].dtype

        # constant data variables
        Ds1, Ds2 = Ds
        Ds_sqr = (
            torch.einsum("ij,il->ijl", Ds1, Ds1).reshape(
                Ds1.shape[0], Ds1.shape[1] ** 2
            ),
            torch.einsum("ij,il->ijl", Ds2, Ds2).reshape(
                Ds2.shape[0], Ds2.shape[1] ** 2
            ),
        )

        Dt1, Dt2 = Dt
        Dt_sqr = (
            torch.einsum("ij,il->ijl", Dt1, Dt1).reshape(
                Dt1.shape[0], Dt1.shape[1] ** 2
            ),
            torch.einsum("ij,il->ijl", Dt2, Dt2).reshape(
                Dt2.shape[0], Dt2.shape[1] ** 2
            ),
        )

        # Same for validation data if provided
        if Ds_val != (None, None) and Dt_val != (None, None):
            Ds1_val, Ds2_val = Ds_val
            Ds_sqr_val = (
                torch.einsum("ij,il->ijl", Ds1_val, Ds1_val).reshape(
                    Ds1_val.shape[0], Ds1_val.shape[1] ** 2
                ),
                torch.einsum("ij,il->ijl", Ds2, Ds2).reshape(
                    Ds2_val.shape[0], Ds2_val.shape[1] ** 2
                ),
            )

            Dt1_val, Dt2_val = Dt_val
            Dt_sqr_val = (
                torch.einsum("ij,il->ijl", Dt1_val, Dt1_val).reshape(
                    Dt1_val.shape[0], Dt1_val.shape[1] ** 2
                ),
                torch.einsum("ij,il->ijl", Dt2_val, Dt2_val).reshape(
                    Dt2_val.shape[0], Dt2_val.shape[1] ** 2
                ),
            )

        else:
            Ds_val, Dt_val = Ds, Dt
            Ds_sqr_val, Dt_sqr_val = Ds_sqr, Dt_sqr

        if alpha == 1 or F[0] is None or F[1] is None:
            alpha = 1
            F = (None, None)

        # initialise coupling and dual vectors
        if init_plan is not None:
            pi = init_plan
            gamma = pi
        else:
            pi = (
                torch.sparse_coo_tensor(
                    torch.from_numpy(
                        np.array(
                            [
                                np.tile(np.arange(n), m),
                                np.repeat(np.arange(m), n),
                            ]
                        )
                    ).type(dtype),
                    torch.from_numpy(np.ones(n * m) / (n * m)).type(dtype),
                    (n, m),
                )
                .to(device)
                .to_sparse_csr()
            )
            gamma = pi

        # measures on rows and columns
        if ws is None:
            ws = torch.ones(n).to(device).to(dtype) / n
        if wt is None:
            wt = torch.ones(m).to(device).to(dtype) / m

        crow_indices, col_indices = pi.crow_indices(), pi.col_indices()
        row_indices = crow_indices_to_row_indices(crow_indices)
        ws_dot_wt_values = ws[row_indices] * wt[col_indices]

        ws_dot_wt = _make_csr_matrix(
            crow_indices, col_indices, ws_dot_wt_values, pi.size(), device
        )

        if solver == "sinkhorn":
            if init_duals is None:
                duals_pi = (
                    torch.zeros_like(ws),
                    torch.zeros_like(wt),
                )
            else:
                duals_pi = init_duals
            duals_gamma = duals_pi
        elif solver == "mm":
            duals_pi, duals_gamma = None, None
        elif solver == "ibpp":
            if init_duals is None:
                duals_pi = (
                    torch.ones_like(ws),
                    torch.ones_like(wt),
                )
            else:
                duals_pi = init_duals
            duals_gamma = duals_pi

        compute_local_biconvex_cost = partial(
            self.local_biconvex_cost,
            data_const=(Ds_sqr, Dt_sqr, Ds, Dt, F),
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, alpha, reg_mode, divergence),
        )

        compute_fugw_loss = partial(
            self.fugw_loss,
            data_const=(Ds_sqr, Dt_sqr, Ds, Dt, F),
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, alpha, reg_mode, divergence),
        )

        compute_fugw_loss_validation = partial(
            self.fugw_loss,
            data_const=(Ds_sqr_val, Dt_sqr_val, Ds_val, Dt_val, F_val),
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, alpha, reg_mode, divergence),
        )

        # If divergence is L2
        self_solver_mm_l2 = partial(
            solver_mm_l2_sparse,
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
        )

        self_get_params_uot_l2 = partial(
            self.get_parameters_uot_l2,
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, reg_mode),
        )

        # If divergence is KL
        self_solver_sinkhorn = partial(
            solver_sinkhorn_sparse,
            tuple_weights=(ws, wt, ws_dot_wt),
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
            verbose=verbose,
        )

        self_solver_mm = partial(
            solver_mm_sparse,
            tuple_weights=(ws, wt),
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
            verbose=verbose,
        )

        self_solver_ibpp = partial(
            solver_ibpp_sparse,
            tuple_weights=(ws, wt, ws_dot_wt),
            train_params=(
                self.nits_uot,
                self.ibpp_nits_sinkhorn,
                self.ibpp_eps_base,
                self.tol_uot,
                self.eval_uot,
            ),
            verbose=verbose,
        )

        # Initialise loss
        current_loss = compute_fugw_loss(pi, gamma)
        loss = _add_dict({}, current_loss)

        if F_val != (None, None):
            loss_val = _add_dict({}, compute_fugw_loss_validation(pi, gamma))
        else:
            loss_val = _add_dict({}, current_loss)

        loss_steps = [0]
        loss_times = [0]
        idx = 0

        # Track difference between two consecutive plans and losses
        # to potentially stop early
        pi_diff = None
        loss_diff = None

        # Run block coordinate descent (BCD) iterations
        t0 = time.time()
        while (
            (pi_diff is None or pi_diff >= self.tol_bcd)
            and (loss_diff is None or loss_diff >= self.tol_loss)
            and (self.nits_bcd is None or idx < self.nits_bcd)
        ):
            pi_prev = pi.detach().clone()

            # Update gamma
            mass_pi = csr_sum(pi)
            cost_gamma = compute_local_biconvex_cost(pi, transpose=True)

            if divergence == "kl":
                new_rho_s = rho_s * mass_pi
                new_rho_t = rho_t * mass_pi
                new_eps = mass_pi * eps if reg_mode == "joint" else eps
                uot_params = (new_rho_s, new_rho_t, new_eps)

                if solver == "sinkhorn":
                    duals_gamma, gamma = self_solver_sinkhorn(
                        cost_gamma, duals_gamma, uot_params
                    )
                elif solver == "mm":
                    gamma = self_solver_mm(cost_gamma, gamma, uot_params)
                if solver == "ibpp":
                    duals_gamma, gamma = self_solver_ibpp(
                        cost_gamma, gamma, duals_gamma, uot_params
                    )
            elif divergence == "l2":
                tuple_weights, uot_params = self_get_params_uot_l2(pi)
                gamma = self_solver_mm_l2(
                    cost_gamma, gamma, uot_params, tuple_weights
                )

            # Rescale gamma
            gamma_scaling_factor = (mass_pi / csr_sum(gamma)).sqrt()
            new_gamma_values = gamma.values() * gamma_scaling_factor
            gamma = torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                new_gamma_values,
                size=gamma.size(),
                device=device,
            )

            # Update pi
            mass_gamma = csr_sum(gamma)
            cost_pi = compute_local_biconvex_cost(gamma, transpose=False)

            if divergence == "kl":
                new_rho_s = rho_s * mass_gamma
                new_rho_t = rho_t * mass_gamma
                new_eps = mass_gamma * eps if reg_mode == "joint" else eps
                uot_params = (new_rho_s, new_rho_t, new_eps)

                if solver == "sinkhorn":
                    duals_pi, pi = self_solver_sinkhorn(
                        cost_pi, duals_pi, uot_params
                    )
                elif solver == "mm":
                    pi = self_solver_mm(cost_pi, pi, uot_params)
                elif solver == "ibpp":
                    duals_pi, pi = self_solver_ibpp(
                        cost_pi, pi, duals_pi, uot_params
                    )
            elif divergence == "l2":
                tuple_weights, uot_params = self_get_params_uot_l2(gamma)
                pi = self_solver_mm_l2(cost_pi, pi, uot_params, tuple_weights)

            # Rescale pi
            pi_scaling_factor = (mass_gamma / csr_sum(pi)).sqrt()
            new_pi_values = pi.values() * pi_scaling_factor
            pi = torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                new_pi_values,
                size=pi.size(),
                device=device,
            )

            if idx % self.eval_bcd == 0:
                current_loss = compute_fugw_loss(pi, gamma)
                if F_val != (None, None):
                    current_loss_validation = compute_fugw_loss_validation(
                        pi, gamma
                    )
                else:
                    current_loss_validation = deepcopy(current_loss)

                loss_steps.append(idx + 1)
                loss = _add_dict(loss, current_loss)
                loss_val = _add_dict(loss_val, current_loss_validation)
                loss_times.append(time.time() - t0)

                if verbose:
                    console.log(
                        f"BCD step {idx+1}/{self.nits_bcd}\t"
                        f"FUGW loss:\t{current_loss['total']}"
                    )

                # Update plan difference for potential early stopping
                if self.tol_bcd is not None:
                    pi_diff = (
                        (pi.values() - pi_prev.values()).abs().sum().item()
                    )

                # Update loss difference for potential early stopping
                if self.tol_loss is not None and len(loss["total"]) >= 2:
                    loss_diff = abs(loss["total"][-2] - loss["total"][-1])

            if callback_bcd is not None:
                callback_bcd(locals())

            idx += 1

        if pi.values().isnan().any() or gamma.values().isnan().any():
            console.log("There is NaN in coupling")

        return {
            "pi": pi,
            "gamma": gamma,
            "duals_pi": duals_pi,
            "duals_gamma": duals_gamma,
            "loss": loss,
            "loss_val": loss_val,
            "loss_steps": loss_steps,
            "loss_times": loss_times,
        }
