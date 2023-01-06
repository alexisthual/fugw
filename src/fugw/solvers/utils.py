import torch
from tqdm import tqdm


def csr_dim_sum(inputs, group_indices, n_groups):
    """In a given tensor, computes sum of elements belonging to the same group.
    Taken from https://discuss.pytorch.org/t/sum-over-various-subsets-of-a-tensor/31881/8

    Args:
        inputs: torch.Tensor of size (n, ) whose values will be summed
        group_indices: torch.Tensor of size (n, )
        n_groups: int, total number of groups

    Returns:
        sums: torch.Tensor of size (n_groups)
    """
    n_inputs = inputs.size(0)
    indices = torch.stack(
        (
            group_indices,
            torch.arange(n_inputs, device=group_indices.device),
        )
    )
    values = torch.ones_like(group_indices, dtype=torch.float)
    one_hot = torch.sparse_coo_tensor(
        indices, values, size=(n_groups, n_inputs)
    )
    return torch.sparse.mm(one_hot, inputs.reshape(-1, 1)).to_dense().flatten()


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
    n_elements_per_row = crow_indices[1:] - crow_indices[:-1]
    arange = torch.arange(crow_indices.shape[0] - 1)
    row_indices = torch.repeat_interleave(arange, n_elements_per_row)
    return row_indices


def batch_elementwise_prod_and_sum(
    X1, X2, idx_1, idx_2, axis=1, max_tensor_size=1e8, verbose=False
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

    rg = range(0, m, batch_size)
    rg = (
        tqdm(rg, desc="Element-wise prod and sum (batch)", leave=False)
        if verbose
        else rg
    )

    res = torch.cat(
        [
            (
                X1[idx_1[i : i + batch_size].type(torch.LongTensor), :]
                * X2[idx_2[i : i + batch_size].type(torch.LongTensor), :]
            ).sum(axis)
            for i in rg
        ]
    )

    return res


def solver_scaling(
    cost, init_duals, uot_params, tuple_pxy, train_params, verbose=True
):
    """
    Scaling algorithm (ie Sinkhorn algorithm).
    Code adapted from Séjourné et al 2020:
    https://github.com/thibsej/unbalanced_gromov_wasserstein.
    """

    log_px, log_py, pxy = tuple_pxy
    vx, vy = init_duals
    rho_x, rho_y, eps = uot_params
    niters, tol, eval_freq = train_params

    tau_x = 1 if torch.isinf(rho_x) else rho_x / (rho_x + eps)
    tau_y = 1 if torch.isinf(rho_y) else rho_y / (rho_y + eps)

    range_niters = (
        tqdm(range(niters), desc="Sinkhorn iteration", leave=False)
        if verbose
        else range(niters)
    )

    for idx in range_niters:
        vx_prev, vy_prev = vx.detach().clone(), vy.detach().clone()
        if rho_y == 0:
            vy = torch.zeros_like(vy)
        else:
            vy = -tau_y * ((vx + log_px)[:, None] - cost / eps).logsumexp(
                dim=0
            )

        if rho_x == 0:
            vx = torch.zeros_like(vx)
        else:
            vx = -tau_x * ((vy + log_py)[None, :] - cost / eps).logsumexp(
                dim=1
            )

        if (
            idx % eval_freq == 0
            and max((vx - vx_prev).abs().max(), (vy - vy_prev).abs().max())
            < tol
        ):
            break

    pi = pxy * (vx[:, None] + vy[None, :] - cost / eps).exp()

    return (vx, vy), pi


def solver_mm(
    cost, init_pi, uot_params, tuple_pxy, train_params, verbose=True
):
    """
    Solve (entropic) UOT using the majorization-minimization algorithm.

    Allow epsilon to be 0 but rho_x and rho_y can't be infinity.

    Note that if the parameters are small so that numerically, the exponential
    of negative cost will contain zeros and this serves as sparsification
    of the optimal plan.

    If the parameters are large, then the resulting optimal plan is more dense
    than the one obtained from scaling algorithm.
    But all parameters should not be too small, otherwise the kernel will
    contain too many zeros.  Consequently, the optimal plan will contain NaN
    (because the Kronecker sum of two marginals will eventually contain zeros,
    and divided by zero will result in undesirable coupling).
    """

    niters, tol, eval_freq = train_params
    px, py = tuple_pxy
    rho_x, rho_y, eps = uot_params

    sum_param = rho_x + rho_y + eps
    tau_x, tau_y, r = rho_x / sum_param, rho_y / sum_param, eps / sum_param
    K = (
        px[:, None] ** (tau_x + r)
        * py[None, :] ** (tau_y + r)
        * (-cost / sum_param).exp()
    )

    pi1, pi2, pi = init_pi.sum(1), init_pi.sum(0), init_pi

    range_niters = (
        tqdm(range(niters), desc="MM iteration", leave=False)
        if verbose
        else range(niters)
    )

    for idx in range_niters:
        pi1_old, pi2_old = pi1.detach().clone(), pi2.detach().clone()
        pi = (
            pi ** (tau_x + tau_y)
            / (pi1[:, None] ** tau_x * pi2[None, :] ** tau_y)
            * K
        )
        pi1, pi2 = pi.sum(1), pi.sum(0)

        if (idx % eval_freq == 0) and max(
            (pi1 - pi1_old).abs().max(), (pi2 - pi2_old).abs().max()
        ) < tol:
            break

    return pi


def solver_mm_sparse(
    cost, init_pi, uot_params, tuple_pxy, train_params, verbose=True
):
    """
    Solve (entropic) UOT using the majorization-minimization algorithm.

    Allow epsilon to be 0 but rho_x and rho_y can't be infinity.

    Note that if the parameters are small so that numerically, the exponential
    of negative cost will contain zeros and this serves as sparsification
    of the optimal plan.

    If the parameters are large, then the resulting optimal plan is more dense
    than the one obtained from scaling algorithm.
    But all parameters should not be too small, otherwise the kernel will
    contain too many zeros.  Consequently, the optimal plan will contain NaN
    (because the Kronecker sum of two marginals will eventually contain zeros,
    and divided by zero will result in undesirable coupling).
    """

    niters, tol, eval_freq = train_params
    px, py = tuple_pxy
    rho_x, rho_y, eps = uot_params

    pi1, pi2, pi = (
        csr_sum(init_pi, dim=1),
        csr_sum(init_pi, dim=0),
        init_pi,
    )
    crow_indices, col_indices = pi.crow_indices(), pi.col_indices()
    row_indices = crow_indices_to_row_indices(crow_indices)

    sum_param = rho_x + rho_y + eps
    tau_x, tau_y, r = rho_x / sum_param, rho_y / sum_param, eps / sum_param
    K_values = (
        px[row_indices] ** (tau_x + r)
        * py[col_indices] ** (tau_y + r)
        * (-cost.values() / sum_param).exp()
    )

    range_niters = (
        tqdm(range(niters), desc="MM iteration", leave=False)
        if verbose
        else range(niters)
    )

    for idx in range_niters:
        pi1_old, pi2_old = pi1.detach().clone(), pi2.detach().clone()
        new_pi_values = (
            pi.values() ** (tau_x + tau_y)
            / (pi1[row_indices] ** tau_x * pi2[col_indices] ** tau_y)
            * K_values
        )
        pi = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            new_pi_values,
            size=pi.size(),
        )

        pi1, pi2 = (
            csr_sum(pi, dim=1),
            csr_sum(pi, dim=0),
        )

        if (idx % eval_freq == 0) and max(
            (pi1 - pi1_old).abs().max(), (pi2 - pi2_old).abs().max()
        ) < tol:
            break

    return pi


def solver_dc(
    cost,
    init_pi,
    init_duals,
    uot_params,
    tuple_pxy,
    train_params,
    verbose=True,
):
    niters, nits_sinkhorn, eps_base, tol, eval_freq = train_params
    rho1, rho2, eps = uot_params
    px, py, pxy = tuple_pxy
    u, v = init_duals
    m1, pi = init_pi.sum(1), init_pi

    sum_eps = eps_base + eps
    tau1 = 1 if rho1 == float("inf") else rho1 / (rho1 + sum_eps)
    tau2 = 1 if rho2 == float("inf") else rho2 / (rho2 + sum_eps)

    K = torch.exp(-cost / sum_eps)
    range_niters = (
        tqdm(range(niters), desc="DC iteration", leave=False)
        if verbose
        else range(niters)
    )

    for idx in range_niters:
        m1_prev = m1.detach().clone()

        # IPOT
        G = (
            K * pi
            if (eps_base / sum_eps) == 1
            else K * pi ** (eps_base / sum_eps)
        )
        for _ in range(nits_sinkhorn):
            v = (
                (G.T @ (u * px)) ** (-tau2)
                if rho2 != 0
                else torch.ones_like(v)
            )
            u = (G @ (v * py)) ** (-tau1) if rho1 != 0 else torch.ones_like(u)
        pi = u[:, None] * G * v[None, :]

        # Check stopping criterion
        if idx % eval_freq == 0:
            m1 = pi.sum(1)
            if m1.isnan().any() or m1.isinf().any():
                raise ValueError(
                    "There is NaN in coupling. Please increase eps_base."
                )

            error = (m1 - m1_prev).abs().max().item()
            if error < tol:
                break

    # renormalize couplings
    pi = pi * pxy

    return (u, v), pi


def solver_dc_sparse(
    cost,
    init_pi,
    init_duals,
    uot_params,
    tuple_pxy,
    train_params,
    verbose=True,
):
    niters, nits_sinkhorn, eps_base, tol, eval_freq = train_params
    rho1, rho2, eps = uot_params
    px, py, pxy = tuple_pxy
    u, v = init_duals
    m1, pi = csr_sum(init_pi, dim=1), init_pi
    crow_indices, col_indices = pi.crow_indices(), pi.col_indices()
    row_indices = crow_indices_to_row_indices(crow_indices)

    sum_eps = eps_base + eps
    tau1 = 1 if rho1 == float("inf") else rho1 / (rho1 + sum_eps)
    tau2 = 1 if rho2 == float("inf") else rho2 / (rho2 + sum_eps)

    K_values = torch.exp(-cost.values() / sum_eps)

    range_niters = (
        tqdm(range(niters), desc="DC iteration", leave=False)
        if verbose
        else range(niters)
    )

    for idx in range_niters:
        m1_prev = m1.detach().clone()

        # IPOT
        G = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            K_values * pi.values() ** (eps_base / sum_eps),
            size=pi.size(),
        )
        for _ in range(nits_sinkhorn):
            v = (
                torch.sparse.mm(
                    G.transpose(1, 0), (u * px).reshape(-1, 1)
                ).squeeze()
                ** (-tau2)
                if rho2 != 0
                else torch.ones_like(v)
            )
            u = (
                torch.sparse.mm(G, (v * py).reshape(-1, 1)).squeeze()
                ** (-tau1)
                if rho1 != 0
                else torch.ones_like(u)
            )

        new_pi_values = u[row_indices] * v[col_indices] * G.values()
        pi = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            new_pi_values,
            size=pi.size(),
        )

        # Check stopping criterion
        if idx % eval_freq == 0:
            m1 = csr_sum(pi, dim=1)
            if m1.isnan().any() or m1.isinf().any():
                raise ValueError(
                    "There is NaN in coupling. Please increase dc_eps_base."
                )

            error = (m1 - m1_prev).abs().max().item()
            if error < tol:
                break

    # renormalize couplings
    pi = torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        pi.values() * pxy.values(),
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
