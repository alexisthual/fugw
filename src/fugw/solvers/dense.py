from functools import partial

import time

import torch

from fugw.solvers.utils import (
    BaseSolver,
    compute_unnormalized_kl,
    compute_quad_divergence,
    compute_divergence,
    solver_ibpp,
    solver_mm,
    solver_sinkhorn,
    solver_mm_l2,
)
from fugw.utils import _add_dict, console


class FUGWSolver(BaseSolver):
    """Solver computing dense solutions"""

    def local_biconvex_cost(
        self,
        pi,
        transpose,
        data_const,
        tuple_weights,
        hyperparams,
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
        X_sqr, Y_sqr, X, Y, D = data_const
        if transpose:
            X_sqr, Y_sqr, X, Y = X_sqr.T, Y_sqr.T, X.T, Y.T

        pi1, pi2 = pi.sum(1), pi.sum(0)

        cost = torch.zeros_like(D)

        # Avoid unnecessary calculation of UGW when alpha = 0
        if alpha != 1 and D is not None:
            wasserstein_cost = D / 2
            cost += (1 - alpha) * wasserstein_cost

        # or UOT when alpha = 1
        if alpha != 0:
            A = X_sqr @ pi1
            B = Y_sqr @ pi2
            gromov_wasserstein_cost = (
                A[:, None] + B[None, :] - 2 * X @ pi @ Y.T
            )

            cost += alpha * gromov_wasserstein_cost

        if divergence == "kl":
            if rho_s != float("inf") and rho_s != 0:
                marginal_cost_dim1 = compute_unnormalized_kl(pi1, ws)
                cost += rho_s * marginal_cost_dim1
            if rho_t != float("inf") and rho_t != 0:
                marginal_cost_dim2 = compute_unnormalized_kl(pi2, wt)
                cost += rho_t * marginal_cost_dim2

            if reg_mode == "joint":
                regularized_cost = compute_unnormalized_kl(pi, ws_dot_wt)
                cost += eps * regularized_cost
        elif divergence == "l2":
            # Marginal constraints do not appear in the cost matrix
            # in the L2 case. See calculations.
            pass

        return cost

    def fugw_loss(
        self,
        pi,
        gamma,
        data_const,
        tuple_weights,
        hyperparams,
    ):
        """Compute the FUGW loss and each of its components.

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
        X_sqr, Y_sqr, X, Y, D = data_const

        pi1, pi2 = pi.sum(1), pi.sum(0)
        gamma1, gamma2 = gamma.sum(1), gamma.sum(0)

        loss_wasserstein = None
        loss_gromov_wasserstein = None
        loss_marginal_constraint_dim1 = None
        loss_marginal_constraint_dim2 = None
        loss_regularization = None
        loss = 0

        if alpha != 1 and D is not None:
            loss_wasserstein = ((D * pi).sum() + (D * gamma).sum()) / 2
            loss += (1 - alpha) * loss_wasserstein

        if alpha != 0:
            A = (X_sqr @ gamma1).dot(pi1)
            B = (Y_sqr @ gamma2).dot(pi2)
            C = (X @ gamma @ Y.T) * pi
            loss_gromov_wasserstein = A + B - 2 * C.sum()
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
                loss_regularization = compute_quad_divergence(
                    pi, gamma, ws_dot_wt, ws_dot_wt, divergence
                )
            elif reg_mode == "independent":
                loss_regularization = compute_divergence(
                    pi, ws_dot_wt, divergence
                ) + compute_divergence(gamma, ws_dot_wt, divergence)
            loss += eps * loss_regularization

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

        pi1, pi2 = pi.sum(1), pi.sum(0)
        l2_pi1, l2_pi2 = (pi1**2).sum(), (pi2**2).sum()
        l2_pi = (pi**2).sum()

        weight_ws = pi1.dot(ws) / l2_pi1
        weight_wt = pi2.dot(wt) / l2_pi2
        weight_wst = (
            (pi * ws_dot_wt).sum() / l2_pi if reg_mode == "joint" else 1
        )
        weighted_tuple_weights = (
            weight_ws * ws,
            weight_wt * wt,
            weight_wst * ws_dot_wt,
        )

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
        F=None,
        Ds=None,
        Dt=None,
        F_val=None,
        Ds_val=None,
        Dt_val=None,
        ws=None,
        wt=None,
        init_plan=None,
        init_duals=None,
        solver="sinkhorn",
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
        F: matrix of size n x m.
            Kernel matrix between the source and target training features.
        Ds: matrix of size n x n
        Dt: matrix of size m x m
        F_val: matrix of size n x m, None
            Kernel matrix between the source and target validation features.
        Ds_val: matrix of size n x n, None
        Dt_val: matrix of size m x m, None
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
                pi: torch.Tensor of size n x m
                    Sample matrix.
                gamma: torch.Tensor of size d1 x d2
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
                loss_val: dict of lists
                    Dictionary containing the loss and its unweighted
                    components for each step of the block-coordinate-descent
                    for which the FUGW loss was evaluated on the validation
                    set.
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

        device, dtype = Ds.device, Ds.dtype

        # constant data variables
        Ds_sqr = Ds**2
        Dt_sqr = Dt**2

        # Same for validation data if provided
        if Ds_val is not None and Dt_val is not None:
            Ds_sqr_val = Ds_val**2
            Dt_sqr_val = Dt_val**2

        else:
            Ds_val, Dt_val = Ds, Dt
            Ds_sqr_val, Dt_sqr_val = Ds_sqr, Dt_sqr

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
        if solver == "mm":
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
            solver_mm_l2,
            train_params=(self.nits_uot, self.tol_uot, self.eval_uot),
        )

        self_get_params_uot_l2 = partial(
            self.get_parameters_uot_l2,
            tuple_weights=(ws, wt, ws_dot_wt),
            hyperparams=(rho_s, rho_t, eps, reg_mode),
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
        current_loss = compute_fugw_loss(pi, gamma)

        if F_val is not None:
            current_loss_validation = compute_fugw_loss_validation(pi, gamma)
        else:
            current_loss_validation = current_loss

        loss = _add_dict({}, current_loss)
        loss_val = _add_dict({}, current_loss_validation)
        loss_steps = [0]
        loss_times = [0]
        idx = 0

        pi_diff = None
        loss_diff = None

        t0 = time.time()
        while (
            (pi_diff is None or pi_diff >= self.tol_bcd)
            and (loss_diff is None or loss_diff >= self.tol_loss)
            and (self.nits_bcd is None or idx < self.nits_bcd)
        ):
            pi_prev = pi.detach().clone()

            # Update gamma
            mass_pi = pi.sum()
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
                    gamma = self_solver_mm_kl(cost_gamma, gamma, uot_params)
                elif solver == "ibpp":
                    duals_gamma, gamma = self_solver_ibpp(
                        cost_gamma, gamma, duals_gamma, uot_params
                    )

            elif divergence == "l2":
                tuple_weights, uot_params = self_get_params_uot_l2(pi)
                gamma = self_solver_mm_l2(
                    cost_gamma, gamma, uot_params, tuple_weights
                )

            # Rescale gamma
            gamma = (mass_pi / gamma.sum()).sqrt() * gamma

            # Update pi
            mass_gamma = gamma.sum()
            cost_pi = compute_local_biconvex_cost(gamma, transpose=False)

            if divergence == "kl":
                new_rho_s, new_rho_t = rho_s * mass_gamma, rho_t * mass_gamma
                new_eps = mass_gamma * eps if reg_mode == "joint" else eps
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
                tuple_weights, uot_params = self_get_params_uot_l2(gamma)
                pi = self_solver_mm_l2(cost_pi, pi, uot_params, tuple_weights)

            # Rescale mass
            pi = (mass_gamma / pi.sum()).sqrt() * pi

            if idx % self.eval_bcd == 0:
                current_loss = compute_fugw_loss(pi, gamma)
                if F_val is not None:
                    current_loss_validation = compute_fugw_loss_validation(
                        pi, gamma
                    )
                else:
                    current_loss_validation = current_loss

                loss_steps.append(idx + 1)
                loss = _add_dict(loss, current_loss)
                loss_val = _add_dict(loss_val, current_loss_validation)
                loss_times.append(time.time() - t0)

                if verbose:
                    console.log(
                        f"BCD step {idx+1}/{self.nits_bcd}\t"
                        f"FUGW loss:\t{current_loss['total']}\t"
                        f"Validation loss:\t{current_loss_validation['total']}"
                    )

                # Update plan difference for potential early stopping
                if self.tol_bcd is not None:
                    pi_diff = (pi - pi_prev).abs().sum().item()

                # Update loss difference for potential early stopping
                if self.tol_loss is not None and len(loss["total"]) >= 2:
                    loss_diff = abs(loss["total"][-2] - loss["total"][-1])

            if callback_bcd is not None:
                callback_bcd(locals())

            idx += 1

        if pi.isnan().any() or gamma.isnan().any():
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
