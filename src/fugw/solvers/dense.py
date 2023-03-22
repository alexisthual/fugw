from functools import partial

import time

import torch

from fugw.solvers.utils import (
    BaseSolver,
    compute_approx_kl,
    compute_quad_divergence,
    compute_divergence,
    solver_ibpp,
    solver_mm,
    solver_sinkhorn,
    solver_mm_l2,
)
from fugw.utils import console


class FUGWSolver(BaseSolver):
    """Solver computing dense solutions"""

    def local_biconvex_cost(
        self, pi, transpose, data_const, tuple_weights, hyperparams, divergence
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
        X_sqr, Y_sqr, X, Y, D = data_const
        if transpose:
            X_sqr, Y_sqr, X, Y = X_sqr.T, Y_sqr.T, X.T, Y.T

        pi1, pi2 = pi.sum(1), pi.sum(0)

        cost = torch.zeros_like(D)

        # Avoid unnecessary calculation of UGW when alpha = 0
        if alpha != 1 and D is not None:
            wasserstein_cost = D
            cost += (1 - alpha) / 2 * wasserstein_cost

        # or UOT when alpha = 1
        if alpha != 0:
            A = X_sqr @ pi1
            B = Y_sqr @ pi2
            gromov_wasserstein_cost = (
                A[:, None] + B[None, :] - 2 * X @ pi @ Y.T
            )

            cost += alpha * gromov_wasserstein_cost

        if divergence == "kl":
            # or when cost is balanced
            if rho_s != float("inf") and rho_s != 0:
                marginal_cost_dim1 = compute_approx_kl(pi1, ws)
                cost += rho_s * marginal_cost_dim1
            if rho_t != float("inf") and rho_t != 0:
                marginal_cost_dim2 = compute_approx_kl(pi2, wt)
                cost += rho_t * marginal_cost_dim2

            if reg_mode == "joint":
                entropic_cost = compute_approx_kl(pi, ws_dot_wt)
                cost += eps * entropic_cost

        return cost

    def fugw_loss(
        self, pi, gamma, data_const, tuple_weights, hyperparams, divergence
    ):
        """
        Returns scalar fugw loss, which is a combination of:
        - a Wasserstein loss on features
        - a Gromow-Wasserstein loss on geometries
        - marginal constraints on the computed OT plan
        - an entropic or L2 regularization
        """

        rho_s, rho_t, eps, alpha, reg_mode = hyperparams
        ws, wt, ws_dot_wt = tuple_weights
        X_sqr, Y_sqr, X, Y, D = data_const

        pi1, pi2 = pi.sum(1), pi.sum(0)
        gamma1, gamma2 = gamma.sum(1), gamma.sum(0)

        loss = 0

        if alpha != 1 and D is not None:
            wasserstein_loss = (D * pi).sum() + (D * gamma).sum()
            loss += (1 - alpha) / 2 * wasserstein_loss

        if alpha != 0:
            A = (X_sqr @ gamma1).dot(pi1)
            B = (Y_sqr @ gamma2).dot(pi2)
            C = (X @ gamma @ Y.T) * pi
            gromov_wasserstein_loss = A + B - 2 * C.sum()
            loss += alpha * gromov_wasserstein_loss

        if rho_s != float("inf") and rho_s != 0:
            marginal_constraint_dim1 = compute_quad_divergence(
                pi1, gamma1, ws, ws, divergence
            )
            loss += rho_s * marginal_constraint_dim1
        if rho_t != float("inf") and rho_t != 0:
            marginal_constraint_dim2 = compute_quad_divergence(
                pi2, gamma2, wt, wt, divergence
            )
            loss += rho_t * marginal_constraint_dim2

        regularized_loss = loss
        if eps != 0:
            if reg_mode == "joint":
                regularization = compute_quad_divergence(
                    pi, gamma, ws_dot_wt, ws_dot_wt, divergence
                )
            elif reg_mode == "independent":
                regularization = compute_divergence(
                    pi, ws_dot_wt, divergence
                ) + compute_divergence(gamma, ws_dot_wt, divergence)
            regularized_loss += eps * regularization

        return loss.item(), regularized_loss.item()

    def get_parameters_uot(self, pi, tuple_weights, hyperparams):
        rho_s, rho_t, eps, reg_mode = hyperparams
        ws, wt, ws_dot_wt = tuple_weights

        pi1, pi2 = pi.sum(1), pi.sum(0)
        l2_pi1, l2_pi2 = (pi1**2).sum(), (pi2**2).sum()
        l2_pi = (pi**2).sum()

        weight_ws = pi1.dot(ws) / l2_pi1
        weight_wt = pi2.dot(wt) / l2_pi2
        weight_wst = (pi * ws_dot_wt).sum() / \
            l2_pi if reg_mode == "joint" else 1
        weighted_tuple_weights = (weight_ws * ws,
                                  weight_wt * wt,
                                  weight_wst * ws_dot_wt)

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
        F=None,
        Ds=None,
        Dt=None,
        ws=None,
        wt=None,
        init_plan=None,
        init_duals=None,
        solver="sinkhorn",
        divergence="kl",
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
        F: matrix of size n x m.
            Kernel matrix between the source and target features.
        Ds: matrix of size n x n
        Dt: matrix of size m x m
        ws: ndarray(n), None
            Measures assigned to source points.
        wt: ndarray(m), None
            Measures assigned to target points.
        init_plan: matrix of size n x m if not None.
            Initialization matrix for coupling.
        init_duals: tuple or None
            Initialization duals for coupling.
        solver: "sinkhorn", "mm", "ibpp"
            Solver to use.
        verbose: bool, optional, defaults to False
            Log solving process.

        Returns
        -------
        res: dict
            Dictionary containing the following keys:
                pi: torch.Tensor of size n x m
                    Sample matrix.
                gamma: torch.Tensor of size d1 x d2
                    Feature matrix.
                duals_pi: tuple of torch.Tensor of size
                    Duals of pi
                duals_gamma: tuple of torch.Tensor of size
                    Duals of gamma
                loss_steps: list
                    BCD steps at the end of which the FUGW loss was evaluated
                loss: list
                    Values of FUGW loss
                loss_regularized: list
                    Values of FUGW loss with entropy or L2 regularization
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

        device, dtype = Ds.device, Ds.dtype

        # constant data variables
        Ds_sqr = Ds**2
        Dt_sqr = Dt**2

        if alpha == 1 or F is None:
            alpha = 1
            F = None

        # measures on rows and columns
        if ws is None:
            n = Ds.shape[0]
            ws = torch.ones(n).to(device).to(dtype) / n
        if wt is None:
            m = Dt.shape[0]
            wt = torch.ones(m).to(device).to(dtype) / m
        ws_dot_wt = ws[:, None] * wt[None, :]

        # initialization coupling
        pi = ws_dot_wt if init_plan is None else init_plan
        gamma = pi

        # initialization of dual vectors
        if divergence == "l2" or solver == "mm":
            duals_pi, duals_gamma = None, None
        else:
            if init_duals is None:
                if solver == "sinkhorn":
                    duals_pi = (
                        torch.zeros_like(ws),
                        torch.zeros_like(wt),
                    )
                elif solver == "ibpp":
                    duals_pi = (
                        torch.ones_like(ws),
                        torch.ones_like(wt),
                    )
            else:
                duals_pi = init_duals
            duals_gamma = duals_pi

        # Global shortcuts
        compute_local_biconvex_cost = partial(
            self.local_biconvex_cost,
            data_const=(Ds_sqr, Dt_sqr, Ds, Dt, F),
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, alpha, reg_mode),
            divergence=divergence,
        )

        compute_fugw_loss = partial(
            self.fugw_loss,
            data_const=(Ds_sqr, Dt_sqr, Ds, Dt, F),
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, alpha, reg_mode),
            divergence=divergence,
        )

        # If divergence is L2
        self_solver_mm_l2 = partial(
            solver_mm_l2,
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
        )

        self_get_params_uot = partial(
            self.get_parameters_uot,
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, reg_mode)
        )

        # If divergence is KL
        self_solver_sinkhorn = partial(
            solver_sinkhorn,
            tuple_weights=(ws, wt, ws_dot_wt),
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
        )

        self_solver_mm_kl = partial(
            solver_mm,
            tuple_weights=(ws, wt),
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
        )

        self_solver_ibpp = partial(
            solver_ibpp,
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

        # Initialize loss
        current_loss, current_loss_regularized = compute_fugw_loss(pi, gamma)
        loss_steps = [0]
        loss = [current_loss]
        loss_regularized = [current_loss_regularized]
        loss_times = [0]
        idx = 0
        err = None

        t0 = time.time()
        while (err is None or err > self.tol_bcd) and (idx < self.nits_bcd):
            pi_prev = pi.detach().clone()

            # Update gamma
            l1_pi = pi.sum()
            cost_gamma = compute_local_biconvex_cost(pi, transpose=True)

            if divergence == "kl":
                new_rho_s, new_rho_t = rho_s * l1_pi, rho_t * l1_pi
                new_eps = l1_pi * eps if reg_mode == "joint" else eps
                uot_params = (new_rho_s, new_rho_t, new_eps)

                if solver == "sinkhorn":
                    duals_gamma, gamma = self_solver_sinkhorn(
                        cost_gamma, duals_gamma, uot_params
                    )
                elif solver == "mm":
                    gamma = self_solver_mm_kl(cost_gamma, gamma, uot_params)
                elif solver == "ibpp":
                    duals_gamma, gamma = self_solver_ibpp(
                        cost_gamma, gamma, duals_gamma, uot_params
                    )

            elif divergence == "l2":
                tuple_weights, uot_params = self_get_params_uot(pi)
                gamma = self_solver_mm_l2(
                    cost_gamma, gamma, uot_params, tuple_weights
                )

            # Rescale mass
            gamma = (l1_pi / gamma.sum()).sqrt() * gamma

            # Update pi
            l1_gamma = gamma.sum()
            cost_pi = compute_local_biconvex_cost(gamma, transpose=False)

            if divergence == "kl":
                new_rho_s, new_rho_t = rho_s * l1_gamma, rho_t * l1_gamma
                new_eps = l1_gamma * eps if reg_mode == "joint" else eps
                uot_params = (new_rho_s, new_rho_t, new_eps)

                if solver == "sinkhorn":
                    duals_pi, pi = self_solver_sinkhorn(
                        cost_pi, duals_pi, uot_params
                    )
                elif solver == "mm":
                    pi = self_solver_mm_kl(cost_pi, pi, uot_params)
                elif solver == "ibpp":
                    duals_pi, pi = self_solver_ibpp(
                        cost_pi, pi, duals_pi, uot_params
                    )

            elif divergence == "l2":
                tuple_weights, uot_params = self_get_params_uot(gamma)
                pi = self_solver_mm_l2(cost_pi, pi, uot_params, tuple_weights)

            # Rescale mass
            pi = (l1_gamma / pi.sum()).sqrt() * pi

            # Update error
            err = (pi - pi_prev).abs().sum().item()
            if idx % self.eval_bcd == 0:
                current_loss, current_loss_regularized = compute_fugw_loss(
                    pi, gamma
                )

                loss_steps.append(idx + 1)
                loss.append(current_loss)
                loss_regularized.append(current_loss_regularized)
                loss_times.append(time.time() - t0)

                if verbose:
                    console.log(
                        f"BCD step {idx+1}/{self.nits_bcd}\t"
                        f"FUGW loss:\t{current_loss} (base)\t"
                        f"{current_loss_regularized} (regularized)"
                    )

                if (
                    len(loss_regularized) >= 2
                    and abs(loss_regularized[-2] - loss_regularized[-1])
                    < self.early_stopping_threshold
                ):
                    break

            idx += 1

        if pi.isnan().any() or gamma.isnan().any():
            console.log("There is NaN in coupling")

        return {
            "pi": pi,
            "gamma": gamma,
            "duals_pi": duals_pi,
            "duals_gamma": duals_gamma,
            "loss_steps": loss_steps,
            "loss": loss,
            "loss_entropic": loss_regularized,
            "loss_times": loss_times,
        }
