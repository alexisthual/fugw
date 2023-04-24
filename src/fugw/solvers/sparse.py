from functools import partial

import time

import numpy as np
import torch

from fugw.solvers.utils import (
    BaseSolver,
    batch_elementwise_prod_and_sum,
    compute_approx_kl,
    compute_approx_kl_sparse,
    compute_kl_sparse,
    compute_quad_kl,
    compute_quad_kl_sparse,
    crow_indices_to_row_indices,
    csr_sum,
    elementwise_prod_fact_sparse,
    solver_sinkhorn_sparse,
    solver_ibpp_sparse,
    solver_mm_sparse,
)
from fugw.utils import add_dict, console, make_csr_matrix


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
            loss_wasserstein = csr_sum(
                elementwise_prod_fact_sparse(K1, K2, pi + gamma)
            )
            loss += (1 - alpha) / 2 * loss_wasserstein

        if alpha != 0:
            A = (Ds_sqr_1 @ (Ds_sqr_2.T @ gamma1)).dot(pi1)
            B = (Dt_sqr_1 @ (Dt_sqr_2.T @ gamma2)).dot(pi2)
            C = elementwise_prod_fact_sparse(
                Ds1, ((Ds2.T @ torch.sparse.mm(gamma, Dt2)) @ Dt1.T).T, pi
            )
            loss_gromov_wasserstein = A + B - 2 * csr_sum(C)
            loss += alpha * loss_gromov_wasserstein

        if rho_s != float("inf") and rho_s != 0:
            loss_marginal_constraint_dim1 = compute_quad_kl(
                pi1, gamma1, ws, ws
            )
            loss += rho_s * loss_marginal_constraint_dim1
        if rho_t != float("inf") and rho_t != 0:
            loss_marginal_constraint_dim2 = compute_quad_kl(
                pi2, gamma2, wt, wt
            )
            loss += rho_t * loss_marginal_constraint_dim2

        if eps != 0:
            if reg_mode == "joint":
                loss_regularization = compute_quad_kl_sparse(
                    pi, gamma, ws_dot_wt, ws_dot_wt
                )
            elif reg_mode == "independent":
                loss_regularization = compute_kl_sparse(
                    pi, ws_dot_wt
                ) + compute_kl_sparse(gamma, ws_dot_wt)

            loss = loss + eps * loss_regularization

        return {
            "wasserstein": loss_wasserstein.item(),
            "gromov_wasserstein": loss_gromov_wasserstein.item(),
            "marginal_constraint_dim1": loss_marginal_constraint_dim1.item(),
            "marginal_constraint_dim2": loss_marginal_constraint_dim2.item(),
            "regularization": loss_regularization.item(),
            "total": loss.item(),
        }

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
        solver="ibpp",
        verbose=False,
    ):
        """Run BCD iterations.

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
        solver: "sinkhorn", "mm", "ibpp"
            Solver to use.
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
            hyperparams=(rho_s, rho_t, eps, alpha, reg_mode),
        )

        compute_fugw_loss = partial(
            self.fugw_loss,
            data_const=(Ds_sqr, Dt_sqr, Ds, Dt, F),
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, alpha, reg_mode),
        )

        self_solver_sinkhorn = partial(
            solver_sinkhorn_sparse,
            tuple_weights=(ws, wt, ws_dot_wt),
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
        )

        self_solver_mm = partial(
            solver_mm_sparse,
            tuple_weights=(ws, wt),
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
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
        loss = add_dict({}, current_loss)
        loss_steps = [0]
        loss_times = [0]
        idx = 0
        err = self.tol_bcd + 1e-3

        t0 = time.time()
        while (err > self.tol_bcd) and (idx < self.nits_bcd):
            pi_prev = pi.detach().clone()

            # Update gamma
            mp = csr_sum(pi)
            new_rho_s = rho_s * mp
            new_rho_t = rho_t * mp
            new_eps = mp * eps if reg_mode == "joint" else eps
            uot_params = (new_rho_s, new_rho_t, new_eps)

            cost_gamma = compute_local_biconvex_cost(pi, transpose=True)
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
                current_loss = compute_fugw_loss(pi, gamma)

                loss_steps.append(idx + 1)
                loss = add_dict(loss, current_loss)
                loss_times.append(time.time() - t0)

                if verbose:
                    console.log(
                        f"BCD step {idx+1}/{self.nits_bcd}\t"
                        f"FUGW loss:\t{current_loss['total']}"
                    )

                if (
                    len(loss["total"]) >= 2
                    and abs(loss["total"][-2] - loss["total"][-1])
                    < self.early_stopping_threshold
                ):
                    break

            idx += 1

        if pi.values().isnan().any() or gamma.values().isnan().any():
            console.log("There is NaN in coupling")

        return {
            "pi": pi,
            "gamma": gamma,
            "duals_pi": duals_pi,
            "duals_gamma": duals_gamma,
            "loss": loss,
            "loss_steps": loss_steps,
            "loss_times": loss_times,
        }
