from functools import partial

import numpy as np
import torch

from fugw.utils import BaseSolver, console, make_csr_matrix
from .utils import (
    batch_elementwise_prod_and_sum,
    compute_approx_kl,
    compute_approx_kl_sparse,
    compute_kl_sparse,
    compute_quad_kl,
    compute_quad_kl_sparse,
    crow_indices_to_row_indices,
    csr_sum,
    elementwise_prod_fact_sparse,
    solver_dc_sparse,
    solver_mm_sparse,
)


class FUGWSparseSolver(BaseSolver):
    def local_biconvex_cost(
        self, pi, transpose, data_const, tuple_weights, hyperparams
    ):
        """
        Before each block coordinate descent (BCD) step,
        the local cost matrix is updated.
        This local cost is a matrix of size (n, m)
        which evaluates the cost between every pair of points
        of the source and target distributions.
        Then, we run a BCD (sinkhorn, dc or mm) step
        which makes use of this cost to update the transport plans.
        """

        rho_s, rho_t, eps, alpha, reg_mode = hyperparams
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
            wasserstein_cost_values = batch_elementwise_prod_and_sum(
                K1, K2, row_indices, col_indices, 1
            )
            cost_values += (1 - alpha) / 2 * wasserstein_cost_values

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

        # or when cost is balanced
        if rho_s != float("inf") and rho_s != 0:
            marginal_cost_dim1 = compute_approx_kl(pi1, ws)
            cost_values += rho_s * marginal_cost_dim1
        if rho_t != float("inf") and rho_t != 0:
            marginal_cost_dim2 = compute_approx_kl(pi2, wt)
            cost_values += rho_t * marginal_cost_dim2

        if reg_mode == "joint":
            cost_values += eps * compute_approx_kl_sparse(pi, ws_dot_wt)

        cost = torch.sparse_csr_tensor(
            crow_indices, col_indices, cost_values, size=pi.size()
        )

        return cost

    def fugw_loss(self, pi, gamma, data_const, tuple_weights, hyperparams):
        """
        Returns a scalar which is a lower bound on the fugw loss.
        This lower bound is a combination of:
        - a Wasserstein loss on features
        - a Gromow-Wasserstein loss on geometries
        - marginal constraints on the computed OT plan
        - an entropic regularisation
        """

        rho_s, rho_t, eps, alpha, reg_mode = hyperparams
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

        loss = 0

        if alpha != 1 and K1 is not None and K2 is not None:
            # TODO: warning, torch sparse transforms LongTensor
            # into IntTensor for crow and col indices
            # when adding 2 CSR matrices together.
            # This could become problematic for sparse matrices
            # with more non-null elements than an int can store.
            wasserstein_loss = csr_sum(
                elementwise_prod_fact_sparse(K1, K2, pi + gamma)
            )
            loss += (1 - alpha) / 2 * wasserstein_loss

        if alpha != 0:
            A = (Ds_sqr_1 @ (Ds_sqr_2.T @ gamma1)).dot(pi1)
            B = (Dt_sqr_1 @ (Dt_sqr_2.T @ gamma2)).dot(pi2)
            C = elementwise_prod_fact_sparse(
                Ds1, ((Ds2.T @ torch.sparse.mm(gamma, Dt2)) @ Dt1.T).T, pi
            )
            gromov_wasserstein_loss = A + B - 2 * csr_sum(C)
            loss += alpha * gromov_wasserstein_loss

        if rho_s != float("inf") and rho_s != 0:
            marginal_constraint_dim1 = compute_quad_kl(pi1, gamma1, ws, ws)
            loss += rho_s * marginal_constraint_dim1
        if rho_t != float("inf") and rho_t != 0:
            marginal_constraint_dim2 = compute_quad_kl(pi2, gamma2, wt, wt)
            loss += rho_t * marginal_constraint_dim2

        if reg_mode == "joint":
            entropic_regularization = compute_quad_kl_sparse(
                pi, gamma, ws_dot_wt, ws_dot_wt
            )
        elif reg_mode == "independent":
            entropic_regularization = compute_kl_sparse(
                pi, ws_dot_wt
            ) + compute_kl_sparse(gamma, ws_dot_wt)

        entropic_loss = loss + eps * entropic_regularization

        return loss.item(), entropic_loss.item()

    def solve(
        self,
        alpha=0.5,
        rho_s=1,
        rho_t=1,
        eps=1e-2,
        reg_mode="joint",
        F=(None, None),
        Ds=(None, None),
        Dt=(None, None),
        ws=None,
        wt=None,
        init_plan=None,
        init_duals=None,
        uot_solver="dc",
        verbose=False,
    ):
        """
        Function running the BCD iterations.

        Parameters
        ----------
        alpha: float
        rho_s: float
        rho_t: float
        eps: float
        reg_mode: str
        F: (ndarray(n, d+2), ndarray(m, d+2)) or (None, None)
        Ds: (ndarray(n, k+2), ndarray(n, k+2)), or (None, None)
        Dt: (ndarray(m, k+2), ndarray(m, k+2)), or (None, None)
        ws: ndarray(n), None
            Measures assigned to source points.
        wt: ndarray(m), None
            Measures assigned to target points.
        init_plan: torch.tensor sparse, None
            Initialisation matrix for sample coupling.
        init_duals: torch.tensor sparse, None
            Initialisation matrix for sample coupling.
        uot_solver: "sinkhorn", "mm", "dc"
            Solver to use.
        verbose: bool, optional, defaults to False
            Log solving process.

        Returns
        -------
        pi: matrix of size n x m. Sample matrix.
        gamma: matrix of size d1 x d2. Feature matrix.
        log_cost: if log is True, return a list of loss
            (without taking into account the regularisation term).
        log_ent_cost: if log is True, return a list of entropic loss.
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
        if uot_solver == "mm" and (
            rho_s == float("inf") or rho_t == float("inf")
        ):
            uot_solver = "dc"
        if uot_solver == "sinkhorn" and eps == 0:
            uot_solver = "dc"

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

        ws_dot_wt = make_csr_matrix(
            crow_indices, col_indices, ws_dot_wt_values, pi.size(), device
        )

        if uot_solver == "mm":
            duals_p, duals_g = None, None
        elif uot_solver == "dc":
            if init_duals is None:
                duals_p = (
                    torch.ones_like(ws),
                    torch.ones_like(wt),
                )
            else:
                duals_p = init_duals
            duals_g = duals_p

        compute_local_biconvex_cost = partial(
            self.local_biconvex_cost,
            data_const=(Ds_sqr, Dt_sqr, Ds, Dt, F),
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, alpha, reg_mode),
        )

        compute_fugw_loss = partial(
            self.fugw_loss,
            data_const=(Ds_sqr, Dt_sqr, Ds, Dt, F),
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, alpha, reg_mode),
        )

        self_solver_mm = partial(
            solver_mm_sparse,
            tuple_weights=(ws, wt),
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
        )

        self_solver_dc = partial(
            solver_dc_sparse,
            tuple_weights=(ws, wt, ws_dot_wt),
            train_params=(
                self.nits_uot,
                self.dc_nits_sinkhorn,
                self.dc_eps_base,
                self.tol_uot,
                self.eval_uot,
            ),
            verbose=verbose,
        )

        # Initialise loss
        loss_steps = []
        loss_ = []
        loss_ent_ = []
        idx = 0
        err = self.tol_bcd + 1e-3

        while (err > self.tol_bcd) and (idx <= self.nits_bcd):
            pi_prev = pi.detach().clone()

            # Update gamma
            mp = csr_sum(pi)
            new_rho_s = rho_s * mp
            new_rho_t = rho_t * mp
            new_eps = mp * eps if reg_mode == "joint" else eps
            uot_params = (new_rho_s, new_rho_t, new_eps)

            cost_gamma = compute_local_biconvex_cost(pi, transpose=True)
            if uot_solver == "mm":
                gamma = self_solver_mm(cost_gamma, gamma, uot_params)
            if uot_solver == "dc":
                duals_g, gamma = self_solver_dc(
                    cost_gamma, gamma, duals_g, uot_params
                )

            gamma_scaling_factor = (mp / csr_sum(gamma)).sqrt()
            gamma = torch.sparse_csr_tensor(
                gamma.crow_indices(),
                gamma.col_indices(),
                gamma.values() * gamma_scaling_factor,
                size=gamma.size(),
                device=device,
            )

            # Update pi
            mg = csr_sum(gamma)
            new_rho_s = rho_s * mg
            new_rho_t = rho_t * mg
            new_eps = mp * eps if reg_mode == "joint" else eps
            uot_params = (new_rho_s, new_rho_t, new_eps)

            cost_pi = compute_local_biconvex_cost(gamma, transpose=False)
            if uot_solver == "mm":
                pi = self_solver_mm(cost_pi, pi, uot_params)
            elif uot_solver == "dc":
                duals_p, pi = self_solver_dc(cost_pi, pi, duals_p, uot_params)

            pi_scaling_factor = (mg / csr_sum(pi)).sqrt()
            pi = torch.sparse_csr_tensor(
                pi.crow_indices(),
                pi.col_indices(),
                pi.values() * pi_scaling_factor,
                size=pi.size(),
                device=device,
            )

            # Update error
            err = (pi.values() - pi_prev.values()).abs().sum().item()
            if idx % self.eval_bcd == 0:
                loss, loss_ent = compute_fugw_loss(pi, gamma)

                loss_steps.append(idx)
                loss_.append(loss)
                loss_ent_.append(loss_ent)

                if verbose:
                    console.log(
                        f"BCD step {idx+1}/{self.nits_bcd}\t"
                        f"FUGW loss:\t{loss} (base)\t{loss_ent} (entropic)"
                    )

                if (
                    len(loss_ent_) >= 2
                    and abs(loss_ent_[-2] - loss_ent_[-1])
                    < self.early_stopping_threshold
                ):
                    break

            idx += 1

        if pi.values().isnan().any() or gamma.values().isnan().any():
            console.log("There is NaN in coupling")

        return pi, gamma, duals_p, duals_g, loss_steps, loss_, loss_ent_
