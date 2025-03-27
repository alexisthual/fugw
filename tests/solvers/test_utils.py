import torch
import ot
import pytest
from fugw.solvers.utils import (
    solver_sinkhorn_stabilized_sparse,
    solver_sinkhorn_eps_scaling_sparse,
    solver_sinkhorn_stabilized,
    solver_sinkhorn_eps_scaling,
)


@pytest.mark.parametrize(
    "pot_method, solver",
    [
        ("sinkhorn_stabilized", solver_sinkhorn_stabilized),
        ("sinkhorn_epsilon_scaling", solver_sinkhorn_eps_scaling),
    ],
)
def test_solvers_sinkhorn(pot_method, solver):
    """Test consistence of the dense sinkhorn solvers with POT."""
    ns = 151
    nt = 104
    nf = 10
    eps = 1.0

    niters, tol, eval_freq = train_params = 100, 1e-16, 1

    ws = torch.ones(ns) / ns
    wt = torch.ones(nt) / nt

    source_features = torch.rand(ns, nf)
    target_features = torch.rand(nt, nf)

    cost = torch.cdist(source_features, target_features)

    gamma, log = ot.sinkhorn(
        ws,
        wt,
        cost,
        eps,
        numItermax=niters,
        stopThr=tol,
        method=pot_method,
        print_period=eval_freq,
        log=True,
    )

    # Check the potentials and the transport plan
    (alpha, beta), pi = solver(
        cost,
        ws,
        wt,
        eps,
        numItermax=niters,
        tol=tol,
        eval_freq=eval_freq,
    )

    assert torch.allclose(
        log["alpha"],
        alpha,
    )
    assert torch.allclose(log["beta"], beta)
    assert torch.allclose(gamma, pi)


@pytest.mark.parametrize(
    "pot_method, solver",
    [
        ("sinkhorn_stabilized", solver_sinkhorn_stabilized_sparse),
        (
            "sinkhorn_epsilon_scaling",
            solver_sinkhorn_eps_scaling_sparse,
        ),
    ],
)
def test_solvers_sinkhorn_sparse(pot_method, solver):
    """Test consistence of the sparse sinkhorn solvers with POT."""
    ns = 151
    nt = 104
    nf = 10
    eps = 1.0

    niters, tol, eval_freq = 100, 1e-16, 1

    ws = torch.ones(ns) / ns
    wt = torch.ones(nt) / nt

    source_features = torch.rand(ns, nf)
    target_features = torch.rand(nt, nf)

    cost = torch.cdist(source_features, target_features)

    gamma, log = ot.sinkhorn(
        ws,
        wt,
        cost,
        eps,
        numItermax=niters,
        stopThr=tol,
        method=pot_method,
        print_period=eval_freq,
        log=True,
    )

    # Check the potentials and the transport plan
    (alpha, beta), pi = solver(
        cost.to_sparse_csr(),
        ws,
        wt,
        eps,
        numItermax=niters,
        tol=tol,
        eval_freq=eval_freq,
    )

    assert torch.allclose(
        log["alpha"],
        alpha,
    )
    assert torch.allclose(log["beta"], beta)
    assert torch.allclose(gamma, pi.to_dense(), atol=1e-6)
