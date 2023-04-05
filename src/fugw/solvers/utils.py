import torch

from fugw.utils import get_progress


class BaseSolver:
    def __init__(
        self,
        nits_bcd=10,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=1e-7,
        early_stopping_threshold=1e-6,
        eval_bcd=1,
        eval_uot=10,
        # ibpp-specific parameters
        ibpp_eps_base=1,
        ibpp_nits_sinkhorn=1,
    ):
        """Init FUGW solver.

        Parameters
        ----------
        nits_bcd: int,
            Number of block-coordinate-descent iterations to run
        nits_uot: int,
            Number of solver iteration to run at each BCD iteration
        tol_bcd: float,
            Stop the BCD procedure early if the absolute difference
            between two consecutive transport plans
            under this threshold
        tol_uot: float,
            Stop the BCD procedure early if the absolute difference
            between two consecutive transport plans
            under this threshold
        early_stopping_threshold: float,
            Stop the BCD procedure early if the FUGW loss falls
            under this threshold
        eval_bcd: int,
            During .fit(), at every eval_bcd step:
            1. compute the FUGW loss and store it in an array
            2. consider stopping early
        eval_uot: int,
            During .fit(), at every eval_uot step:
            1. consider stopping early
        ibpp_eps_base: int,
            Regularization parameter specific to the ibpp solver
        ibpp_nits_sinkhorn: int,
            Number of sinkhorn iterations to run
            within each uot iteration of the ibpp solver.
        """

        self.nits_bcd = nits_bcd
        self.nits_uot = nits_uot
        self.tol_bcd = tol_bcd
        self.tol_uot = tol_uot
        self.early_stopping_threshold = early_stopping_threshold
        self.eval_bcd = eval_bcd
        self.eval_uot = eval_uot
        self.ibpp_eps_base = ibpp_eps_base
        self.ibpp_nits_sinkhorn = ibpp_nits_sinkhorn


def csr_dim_sum(values, group_indices, n_groups):
    """
    In a given tensor V, computes sum of elements belonging to the same group.
    It creates a sparse matrix A of size (m, n) where
    m is the number of groups and n the size of V
    such that A_{i,j} equals
    1 if the jth element of V belongs to group i,
    0 otherwise.
    Then it performs an efficient matrix product between A and V.
    See the following discussion for more insights:
    https://discuss.pytorch.org/t/sum-over-various-subsets-of-a-tensor/31881/8

    Args:
        values: torch.Tensor of size (n, ) whose values will be summed
        group_indices: torch.Tensor of size (n, )
        n_groups: int, total number of groups

    Returns:
        sums: torch.Tensor of size (n_groups)
    """
    device = values.device
    n_values = values.size(0)

    # sparse matrix indices (rows, columns)
    indices = torch.stack(
        (
            group_indices,
            torch.arange(n_values).to(device),
        )
    )

    A = torch.sparse_coo_tensor(
        indices,
        torch.ones_like(group_indices).type(torch.float32).to(device),
        size=(n_groups, n_values),
    )

    return torch.sparse.mm(A, values.reshape(-1, 1)).to_dense().flatten()


def csr_sum(csr_matrix, dim=None):
    """Computes sum of a CSR torch matrix along a given dimension."""
    if dim is None:
        return csr_matrix.values().sum()
    elif dim == 0:
        return csr_dim_sum(
            csr_matrix.values(), csr_matrix.col_indices(), csr_matrix.shape[1]
        )
    elif dim == 1:
        row_indices = crow_indices_to_row_indices(csr_matrix.crow_indices())
        return csr_dim_sum(
            csr_matrix.values(), row_indices, csr_matrix.shape[0]
        )
    else:
        raise ValueError(f"Wrong dim: {dim}")


def crow_indices_to_row_indices(crow_indices):
    """
    Computes a row indices tensor
    from a CSR indptr tensor (ie crow indices tensor)
    """
    device = crow_indices.device
    n_elements_per_row = crow_indices[1:] - crow_indices[:-1]
    arange = torch.arange(crow_indices.shape[0] - 1).to(device)
    row_indices = torch.repeat_interleave(arange, n_elements_per_row)
    return row_indices


def fill_csr_matrix_rows(rows, crow_indices):
    """
    Returns values of a CSR matrix M
    such that for all i and j, M[i, j] = rows[i].
    """
    new_values = torch.repeat_interleave(
        rows, crow_indices[1:] - crow_indices[:-1]
    )

    return new_values


def fill_csr_matrix_cols(cols, ccol_indices, csc_to_csr):
    """
    Returns values of a CSR matrix M
    such that for all i and j, M[i, j] = cols[j].
    """
    new_values = torch.repeat_interleave(
        cols, ccol_indices[1:] - ccol_indices[:-1]
    )

    return new_values[csc_to_csr]


def batch_elementwise_prod_and_sum(
    X1, X2, idx_1, idx_2, axis=1, max_tensor_size=1e8
):
    """Batch computation of (X1[idx_1, :] * X2[idx_2, :]).sum(1)

    Parameters
    ----------
    X1: torch.Tensor
        shape(n, d)
    X2: torch.Tensor
        shape(n, d)
    idx_1: torch.Tensor
        shape(m,)
    idx_2: torch.Tensor
        shape(m,)

    Returns
    -------
    result: torch.Tensor
        shape(m,)
    """

    m, d = idx_1.shape[0], X1.shape[1]

    if isinstance(max_tensor_size, (int, float)):
        batch_size = min(int(max_tensor_size / d), m)
    else:
        raise Exception(
            f"Invalid value for max_tensor_size: {max_tensor_size}"
        )

    res = torch.cat(
        [
            (
                X1[idx_1[i : i + batch_size].type(torch.LongTensor), :]  # noqa
                * X2[
                    idx_2[i : i + batch_size].type(torch.LongTensor), :  # noqa
                ]
            ).sum(axis)
            for i in range(0, m, batch_size)
        ]
    )

    return res


def solver_sinkhorn(
    cost, init_duals, uot_params, tuple_weights, train_params, verbose=True
):
    """
    Scaling algorithm (ie Sinkhorn algorithm).
    Code adapted from Séjourné et al 2020:
    https://github.com/thibsej/unbalanced_gromov_wasserstein.
    """

    ws, wt, ws_dot_wt = tuple_weights
    log_ws = ws.log()
    log_wt = wt.log()
    u, v = init_duals
    rho_s, rho_t, eps = uot_params
    niters, tol, eval_freq = train_params

    tau_s = 1 if torch.isinf(rho_s) else rho_s / (rho_s + eps)
    tau_t = 1 if torch.isinf(rho_t) else rho_t / (rho_t + eps)

    with get_progress(transient=True) as progress:
        if verbose:
            task = progress.add_task("Sinkhorn iterations", total=niters)

        for idx in range(niters):
            u_prev, v_prev = u.detach().clone(), v.detach().clone()
            if rho_t == 0:
                v = torch.zeros_like(v)
            else:
                v = -tau_t * ((u + log_ws)[:, None] - cost / eps).logsumexp(
                    dim=0
                )

            if rho_s == 0:
                u = torch.zeros_like(u)
            else:
                u = -tau_s * ((v + log_wt)[None, :] - cost / eps).logsumexp(
                    dim=1
                )

            if verbose:
                progress.update(task, advance=1)

            if (
                idx % eval_freq == 0
                and max((u - u_prev).abs().max(), (v - v_prev).abs().max())
                < tol
            ):
                break

    pi = ws_dot_wt * (u[:, None] + v[None, :] - cost / eps).exp()

    return (u, v), pi


def solver_sinkhorn_sparse(
    cost, init_duals, uot_params, tuple_weights, train_params, verbose=True
):
    """
    Scaling algorithm (ie Sinkhorn algorithm).
    Code adapted from Séjourné et al 2020:
    https://github.com/thibsej/unbalanced_gromov_wasserstein.
    """

    ws, wt, ws_dot_wt = tuple_weights
    log_ws = ws.log()
    log_wt = wt.log()
    u, v = init_duals
    rho_s, rho_t, eps = uot_params
    niters, tol, eval_freq = train_params

    tau_s = 1 if torch.isinf(rho_s) else rho_s / (rho_s + eps)
    tau_t = 1 if torch.isinf(rho_t) else rho_t / (rho_t + eps)

    crow_indices = cost.crow_indices()
    row_indices = crow_indices_to_row_indices(crow_indices)

    # This solver computes sparse matrices with the same
    # sparsity mask as cost
    # whose rows (resp. cols) are constant.
    # To speed-up computations, we pre-compute ccol_indices and csc_to_csr,
    # 2 tensors which allow us to quickly generate such values.
    cost_csc = cost.to_sparse_csc()
    ccol_indices = cost_csc.ccol_indices()
    T = torch.sparse_csc_tensor(
        cost_csc.ccol_indices(),
        cost_csc.row_indices(),
        # Add 1 to arange so that first coefficient is not 0
        torch.arange(cost_csc.values().shape[0]) + 1,
        size=cost_csc.size(),
        device=cost_csc.device,
    ).to_sparse_csr()
    csc_to_csr = T.values() - 1

    with get_progress(transient=True) as progress:
        if verbose:
            task = progress.add_task("Sinkhorn iterations", total=niters)

        for idx in range(niters):
            u_prev, v_prev = u.detach().clone(), v.detach().clone()
            if rho_t == 0:
                v = torch.zeros_like(v)
            else:
                # ((u + log_ws)[:, None] - cost / eps).logsumexp(dim=0)
                rows = u + log_ws
                new_values = fill_csr_matrix_rows(rows, crow_indices)
                new_values = (new_values - (cost.values() / eps)).log()
                csr = torch.sparse_csr_tensor(
                    cost.crow_indices(),
                    cost.col_indices(),
                    new_values,
                    size=cost.size(),
                    device=cost.device,
                )
                v = -tau_t * csr_sum(csr, dim=0).to_dense().exp()

            if rho_s == 0:
                u = torch.zeros_like(u)
            else:
                # ((v + log_wt)[None, :] - cost / eps).logsumexp(dim=1)
                cols = v + log_wt
                new_values = fill_csr_matrix_cols(
                    cols, ccol_indices, csc_to_csr
                )
                new_values = (new_values - (cost.values() / eps)).log()
                new_values = (
                    csr_dim_sum(new_values, row_indices, cost.shape[0])
                    .to_dense()
                    .exp()
                )
                u = -tau_s * new_values

            if verbose:
                progress.update(task, advance=1)

            if (
                idx % eval_freq == 0
                and max((u - u_prev).abs().max(), (v - v_prev).abs().max())
                < tol
            ):
                break

    # pi = ws_dot_wt * (u[:, None] + v[None, :] - cost / eps).exp()
    new_values_pi = (
        ws_dot_wt.values()
        * (
            fill_csr_matrix_rows(u, crow_indices)
            + fill_csr_matrix_cols(v, ccol_indices, csc_to_csr)
            - (cost.values() / eps)
        ).exp()
    )

    pi = torch.sparse_csr_tensor(
        cost.crow_indices(),
        cost.col_indices(),
        new_values_pi,
        size=cost.size(),
    )

    return (u, v), pi


def solver_mm(
    cost, init_pi, uot_params, tuple_weights, train_params, verbose=True
):
    """
    Solve (entropic) UOT using the majorization-minimization algorithm.

    Allow epsilon to be 0 but rho_s and rho_t can't be infinity.

    Note that if the parameters are small so that numerically, the exponential
    of negative cost will contain zeros and this serves as sparsification
    of the optimal plan.

    If the parameters are large, then the resulting optimal plan is more dense
    than the one obtained from sinkhorn algorithm.
    But all parameters should not be too small, otherwise the kernel will
    contain too many zeros. Consequently, the optimal plan will contain NaN
    (because the Kronecker sum of two marginals will eventually contain zeros,
    and divided by zero will result in undesirable coupling).
    """

    niters, tol, eval_freq = train_params
    ws, wt = tuple_weights
    rho_s, rho_t, eps = uot_params

    sum_param = rho_s + rho_t + eps
    tau_s, tau_t, r = rho_s / sum_param, rho_t / sum_param, eps / sum_param
    K = (
        ws[:, None] ** (tau_s + r)
        * wt[None, :] ** (tau_t + r)
        * (-cost / sum_param).exp()
    )

    pi1, pi2, pi = init_pi.sum(1), init_pi.sum(0), init_pi

    with get_progress(transient=True) as progress:
        if verbose:
            task = progress.add_task("MM-KL iterations", total=niters)

        for idx in range(niters):
            pi1_prev, pi2_prev = pi1.detach().clone(), pi2.detach().clone()
            pi = (
                pi ** (tau_s + tau_t)
                / (pi1[:, None] ** tau_s * pi2[None, :] ** tau_t)
                * K
            )
            pi1, pi2 = pi.sum(1), pi.sum(0)

            if verbose:
                progress.update(task, advance=1)

            if idx % eval_freq == 0:
                pi1_error = (pi1 - pi1_prev).abs().max()
                pi2_error = (pi2 - pi2_prev).abs().max()
                if max(pi1_error, pi2_error) < tol:
                    if verbose:
                        progress.console.log(
                            "Reached tol_uot threshold: "
                            f"{pi1_error}, {pi2_error}"
                        )
                    break

    return pi


def solver_mm_l2(
    cost, init_pi, uot_params, tuple_weights, train_params, verbose=True
):
    """
    Solve regularized UOT with L2-squared norm using
    the majorization-minimization algorithm. Allow epsilon to be 0
    but rho_s and rho_t can't be infinity.

    If $\rho$ is too small, then we obtain $0$ everywhere,
    which will result in NaN coupling.
    If $\rho$ is too large, then we lose the sparsity.
    So, need to choose $\rho$ adequately.
    """

    niters, tol, eval_freq = train_params
    ws, wt, ws_dot_wt = tuple_weights
    rho_s, rho_t, eps = uot_params

    thres = (
        rho_s * ws[:, None] + rho_t * wt[None, :] + eps * ws_dot_wt - cost
    )
    thres = torch.clamp(thres, min=0)

    pi1, pi2, pi = init_pi.sum(1), init_pi.sum(0), init_pi

    with get_progress(transient=True) as progress:
        if verbose:
            task = progress.add_task("MM-L2 iterations", total=niters)

        for idx in range(niters):
            pi1_prev, pi2_prev = pi1.detach().clone(), pi2.detach().clone()
            denom = rho_s * pi1[:, None] + rho_t * pi2[None, :] + eps * pi
            pi = thres * pi / denom
            pi1, pi2 = pi.sum(1), pi.sum(0)

            if verbose:
                progress.update(task, advance=1)

            if idx % eval_freq == 0:
                pi1_error = (pi1 - pi1_prev).abs().max()
                pi2_error = (pi2 - pi2_prev).abs().max()
                if max(pi1_error, pi2_error) < tol:
                    if verbose:
                        progress.console.log(
                            "Reached tol_uot threshold: "
                            f"{pi1_error}, {pi2_error}"
                        )
                    break

    return pi


def solver_mm_sparse(
    cost, init_pi, uot_params, tuple_weights, train_params, verbose=True
):
    """
    Solve (entropic) UOT using the majorization-minimization algorithm.

    Allow epsilon to be 0 but rho_s and rho_t can't be infinity.

    Note that if the parameters are small so that numerically, the exponential
    of negative cost will contain zeros and this serves as sparsification
    of the optimal plan.

    If the parameters are large, then the resulting optimal plan is more dense
    than the one obtained from sinkhorn algorithm.
    But all parameters should not be too small, otherwise the kernel will
    contain too many zeros. Consequently, the optimal plan will contain NaN
    (because the Kronecker sum of two marginals will eventually contain zeros,
    and divided by zero will result in undesirable coupling).
    """

    niters, tol, eval_freq = train_params
    ws, wt = tuple_weights
    rho_s, rho_t, eps = uot_params

    pi1, pi2, pi = (
        csr_sum(init_pi, dim=1),
        csr_sum(init_pi, dim=0),
        init_pi,
    )
    crow_indices, col_indices = pi.crow_indices(), pi.col_indices()
    row_indices = crow_indices_to_row_indices(crow_indices)
    pi_values = pi.values()

    sum_param = rho_s + rho_t + eps
    tau_s, tau_t, r = rho_s / sum_param, rho_t / sum_param, eps / sum_param
    K_values = (
        ws[row_indices] ** (tau_s + r)
        * wt[col_indices] ** (tau_t + r)
        * (-cost.values() / sum_param).exp()
    )

    # Define sparse matrices row_one_hot_indices and col_one_hot_indices
    # which are useful for computing row / column sums efficiently.
    # values, group_indices, n_groups
    device = pi_values.device
    n_rows = init_pi.shape[0]
    n_cols = init_pi.shape[1]
    n_pi_values = pi_values.size(0)
    # Sparse matrix to sum on rows
    row_one_hot_indices = torch.stack(
        (
            row_indices,
            torch.arange(n_pi_values).to(device),
        )
    )
    row_one_hot = torch.sparse_coo_tensor(
        row_one_hot_indices,
        torch.ones_like(row_indices).type(torch.float32).to(device),
        size=(n_rows, n_pi_values),
    ).to_sparse_csr()
    # Sparse matrix to sum on columns
    col_one_hot_indices = torch.stack(
        (
            col_indices,
            torch.arange(n_pi_values).to(device),
        )
    )
    col_one_hot = torch.sparse_coo_tensor(
        col_one_hot_indices,
        torch.ones_like(col_indices).type(torch.float32).to(device),
        size=(n_cols, n_pi_values),
    ).to_sparse_csr()

    with get_progress(transient=True) as progress:
        if verbose:
            task = progress.add_task("MM iterations", total=niters)

        for idx in range(niters):
            pi1_prev, pi2_prev = pi1.detach().clone(), pi2.detach().clone()

            pi_values = (
                pi_values ** (tau_s + tau_t)
                / (pi1[row_indices] ** tau_s * pi2[col_indices] ** tau_t)
                * K_values
            )

            # Efficiently compute sum on rows
            pi1 = (
                torch.sparse.mm(row_one_hot, pi_values.reshape(-1, 1))
                .to_dense()
                .flatten()
            )

            # Efficiently compute sum on columns
            pi2 = (
                torch.sparse.mm(col_one_hot, pi_values.reshape(-1, 1))
                .to_dense()
                .flatten()
            )

            if verbose:
                progress.update(task, advance=1)

            if idx % eval_freq == 0:
                pi1_error = (pi1 - pi1_prev).abs().max()
                pi2_error = (pi2 - pi2_prev).abs().max()
                if max(pi1_error, pi2_error) < tol:
                    if verbose:
                        progress.console.log(
                            "Reached tol_uot threshold: "
                            f"{pi1_error}, {pi2_error}"
                        )
                    break

    return torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        pi_values,
        size=pi.size(),
        device=device,
    )


def solver_ibpp(
    cost,
    init_pi,
    init_duals,
    uot_params,
    tuple_weights,
    train_params,
    verbose=True,
):
    niters, nits_sinkhorn, eps_base, tol, eval_freq = train_params
    rho_s, rho_t, eps = uot_params
    ws, wt, ws_dot_wt = tuple_weights
    u, v = init_duals
    m1, pi = init_pi.sum(1), init_pi

    sum_eps = eps_base + eps
    tau_s = 1 if rho_s == float("inf") else rho_s / (rho_s + sum_eps)
    tau_t = 1 if rho_t == float("inf") else rho_t / (rho_t + sum_eps)

    K = torch.exp(-cost / sum_eps)

    with get_progress(transient=True) as progress:
        if verbose:
            task = progress.add_task("DC iterations", total=niters)

        for idx in range(niters):
            m1_prev = m1.detach().clone()

            # IPOT
            G = (
                K * pi
                if (eps_base / sum_eps) == 1
                else K * pi ** (eps_base / sum_eps)
            )
            for _ in range(nits_sinkhorn):
                v = (
                    (G.T @ (u * ws)) ** (-tau_t)
                    if rho_t != 0
                    else torch.ones_like(v)
                )
                u = (
                    (G @ (v * wt)) ** (-tau_s)
                    if rho_s != 0
                    else torch.ones_like(u)
                )
            pi = u[:, None] * G * v[None, :]

            if verbose:
                progress.update(task, advance=1)

            # Check stopping criterion
            if idx % eval_freq == 0:
                m1 = pi.sum(1)
                if m1.isnan().any() or m1.isinf().any():
                    raise ValueError(
                        "There is NaN in coupling. "
                        "You may want to increase ibpp_eps_base "
                        f"(current value: {eps_base})."
                    )

                error = (m1 - m1_prev).abs().max().item()
                if error < tol:
                    if verbose:
                        progress.console.log(
                            f"Reached tol_uot threshold: {error}"
                        )
                    break

    # renormalize couplings
    pi = pi * ws_dot_wt

    return (u, v), pi


def solver_ibpp_sparse(
    cost,
    init_pi,
    init_duals,
    uot_params,
    tuple_weights,
    train_params,
    verbose=True,
):
    niters, nits_sinkhorn, eps_base, tol, eval_freq = train_params
    rho_s, rho_t, eps = uot_params
    ws, wt, ws_dot_wt = tuple_weights
    u, v = init_duals
    m1, pi = csr_sum(init_pi, dim=1), init_pi
    device = pi.device
    crow_indices, col_indices = pi.crow_indices(), pi.col_indices()
    row_indices = crow_indices_to_row_indices(crow_indices)

    sum_eps = eps_base + eps
    tau_s = 1 if rho_s == float("inf") else rho_s / (rho_s + sum_eps)
    tau_t = 1 if rho_t == float("inf") else rho_t / (rho_t + sum_eps)

    K_values = torch.exp(-cost.values() / sum_eps)

    # This solver computes matrix multiplications with a
    # sparse matrix G and it's transpose.
    # To speed-up computation, we compute csr_values_to_transpose_values,
    # a torch array indicating how the values of a CSR matrix A
    # should be reordered so as to form a CSR matrix representing A.transpose()
    T = (
        torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            # Add 1 to arange so that first coefficient is not 0
            torch.arange(pi.values().shape[0]) + 1,
            size=pi.size(),
            device=device,
        )
        .to_sparse_csc()
        .transpose(1, 0)
    )
    # Remove previously added 1
    csr_values_to_transpose_values = T.values() - 1

    with get_progress(transient=True) as progress:
        if verbose:
            task = progress.add_task("DC iterations", total=niters)

        for idx in range(niters):
            m1_prev = m1.detach().clone()

            new_G_values = K_values * pi.values() ** (eps_base / sum_eps)
            G = torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                new_G_values,
                size=pi.size(),
                device=device,
            )
            G_transpose = torch.sparse_csr_tensor(
                T.crow_indices(),
                T.col_indices(),
                new_G_values[csr_values_to_transpose_values],
                size=(pi.size()[1], pi.size()[0]),
                device=device,
            )

            for _ in range(nits_sinkhorn):
                v = (
                    torch.sparse.mm(
                        G_transpose,
                        (u * ws).reshape(-1, 1),
                    ).squeeze()
                    ** (-tau_t)
                    if rho_t != 0
                    else torch.ones_like(v)
                )
                u = (
                    torch.sparse.mm(G, (v * wt).reshape(-1, 1)).squeeze()
                    ** (-tau_s)
                    if rho_s != 0
                    else torch.ones_like(u)
                )

            new_pi_values = u[row_indices] * v[col_indices] * G.values()
            pi = torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                new_pi_values,
                size=pi.size(),
            )

            if verbose:
                progress.update(task, advance=1)

            # Check stopping criterion
            if idx % eval_freq == 0:
                m1 = csr_sum(pi, dim=1)
                if m1.isnan().any() or m1.isinf().any():
                    raise ValueError(
                        "There is NaN in coupling. "
                        "You may want to increase ibpp_eps_base "
                        f"(current value: {eps_base})."
                    )

                error = (m1 - m1_prev).abs().max().item()
                if error < tol:
                    if verbose:
                        progress.console.log(
                            f"Reached tol_uot threshold: {error}"
                        )
                    break

    # renormalize couplings
    pi = torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        pi.values() * ws_dot_wt.values(),
        size=pi.size(),
    )

    return (u, v), pi


def compute_approx_kl(p, q):
    # By convention: 0 log 0 = 0
    entropy = torch.nan_to_num(
        p * (p / q).log(), nan=0.0, posinf=0.0, neginf=0.0
    ).sum()
    return entropy


def elementwise_prod_sparse(p, q):
    """Element-wise product between 2 sparse CSR matrices
    which have the same sparcity indices."""
    values = p.values() * q.values()
    return torch.sparse_csr_tensor(
        p.crow_indices(), p.col_indices(), values, size=p.size()
    )


def elementwise_prod_fact_sparse(a, b, p):
    """Compute (AB * pi) without exceeding memory capacity."""
    crow_indices, col_indices = p.crow_indices(), p.col_indices()
    row_indices = crow_indices_to_row_indices(crow_indices)
    values = batch_elementwise_prod_and_sum(a, b, row_indices, col_indices, 1)
    values = values * p.values()
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, size=p.size()
    )


def compute_approx_kl_sparse(p, q):
    return compute_approx_kl(p.values(), q.values())


def compute_kl(p, q):
    return compute_approx_kl(p, q) - p.sum() + q.sum()


def compute_kl_sparse(p, q):
    return compute_approx_kl_sparse(p, q) - csr_sum(p) + csr_sum(q)


def compute_quad_kl(mu, nu, alpha, beta):
    """
    Calculate the KL divergence between two product measures:
    KL(mu otimes nu, alpha otimes beta) =
    m_mu * KL(nu, beta)
    + m_nu * KL(mu, alpha)
    + (m_mu - m_alpha) * (m_nu - m_beta)

    Parameters
    ----------
    mu: vector or matrix
    nu: vector or matrix
    alpha: vector or matrix with the same size as mu
    beta: vector or matrix with the same size as nu

    Returns
    ----------
    KL divergence between two product measures
    """

    m_mu = mu.sum()
    m_nu = nu.sum()
    m_alpha = alpha.sum()
    m_beta = beta.sum()
    const = (m_mu - m_alpha) * (m_nu - m_beta)
    kl = m_nu * compute_kl(mu, alpha) + m_mu * compute_kl(nu, beta) + const

    return kl


def compute_quad_kl_sparse(mu, nu, alpha, beta):
    m_mu = csr_sum(mu)
    m_nu = csr_sum(nu)
    m_alpha = csr_sum(alpha)
    m_beta = csr_sum(beta)
    const = (m_mu - m_alpha) * (m_nu - m_beta)
    kl = (
        m_nu * compute_kl_sparse(mu, alpha)
        + m_mu * compute_kl_sparse(nu, beta)
        + const
    )

    return kl


def compute_quad_l2(a, b, mu, nu):
    """
    Compute || a otimes b - mu otimes nu ||^2
    """

    norm_a2 = (a**2).sum()
    ratio = (a * mu).sum() / norm_a2
    norm = norm_a2 * torch.sum((b - ratio * nu) ** 2) + (nu**2).sum() * (
        torch.sum((mu**2)) - ratio
    )

    return norm


def compute_l2(p, q):
    return torch.sum((p - q) ** 2)


def compute_quad_divergence(mu, nu, alpha, beta, divergence="kl"):
    if divergence == "kl":
        return compute_quad_kl(mu, nu, alpha, beta)
    elif divergence == "l2":
        return compute_quad_l2(mu, nu, alpha, beta)


def compute_divergence(p, q, divergence="kl"):
    if divergence == "kl":
        return compute_kl(p, q)
    elif divergence == "l2":
        return compute_l2(p, q)
