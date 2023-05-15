from itertools import product

import numpy as np
import pytest
import torch

from fugw.solvers import FUGWSparseSolver
from fugw.utils import _low_rank_squared_l2


devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))


callbacks = [None, lambda x: x["gamma"]]

# TODO: need to test sinkhorn
@pytest.mark.parametrize(
    "solver,device,callback",
    product(["mm", "ibpp"], devices, callbacks),
)
def test_sparse_solvers(solver, device, callback):
    torch.manual_seed(1)
    torch.backends.cudnn.benchmark = True

    ns = 104
    ds = 3
    nt = 151
    dt = 7
    nf = 10

    source_features = torch.rand(ns, nf).to(device)
    target_features = torch.rand(nt, nf).to(device)
    source_embeddings = torch.rand(ns, ds).to(device)
    target_embeddings = torch.rand(nt, dt).to(device)

    F = _low_rank_squared_l2(source_features, target_features)
    Ds = _low_rank_squared_l2(source_embeddings, source_embeddings)
    Dt = _low_rank_squared_l2(target_embeddings, target_embeddings)

    F_norm = (F[0] @ F[1].T).max()
    Ds_norm = (Ds[0] @ Ds[1].T).max()
    Dt_norm = (Dt[0] @ Dt[1].T).max()

    F_normalized = (F[0] / F_norm, F[1] / F_norm)
    Ds_normalized = (Ds[0] / Ds_norm, Ds[1] / Ds_norm)
    Dt_normalized = (Dt[0] / Dt_norm, Dt[1] / Dt_norm)

    init_plan = (
        (torch.ones(ns, nt) / ns).to_sparse_coo().to(device).to_sparse_csr()
    )

    nits_bcd = 100
    eval_bcd = 1
    fugw = FUGWSparseSolver(
        nits_bcd=nits_bcd,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=1e-7,
        early_stopping_threshold=1e-5,
        eval_bcd=eval_bcd,
        eval_uot=10,
        # Set a high value of ibpp, otherwise nans appear in coupling.
        # This will generally increase the computed fugw loss.
        ibpp_eps_base=1e2,
    )

    res = fugw.solve(
        alpha=0.2,
        rho_s=2,
        rho_t=3,
        eps=0.02,
        reg_mode="independent",
        F=F_normalized,
        Ds=Ds_normalized,
        Dt=Dt_normalized,
        init_plan=init_plan,
        solver=solver,
        callback_bcd=callback,
        verbose=True,
    )

    pi = res["pi"]
    gamma = res["gamma"]
    duals_pi = res["duals_pi"]
    duals_gamma = res["duals_gamma"]
    loss = res["loss"]
    loss_steps = res["loss_steps"]
    loss_times = res["loss_times"]

    assert pi.size() == (ns, nt)
    assert gamma.size() == (ns, nt)

    if solver == "mm":
        assert duals_pi is None
        assert duals_gamma is None
    elif solver == "ibpp":
        assert len(duals_pi) == 2
        assert duals_pi[0].shape == (ns,)
        assert duals_pi[1].shape == (nt,)

    assert len(loss_steps) - 1 <= nits_bcd // eval_bcd + 1
    assert len(loss_times) == len(loss_steps)
    for key in [
        "wasserstein",
        "gromov_wasserstein",
        "marginal_constraint_dim1",
        "marginal_constraint_dim2",
        "regularization",
        "total",
    ]:
        assert len(loss[key]) == len(loss_steps)
    # Loss should decrease
    assert np.all(
        np.sign(np.array(loss["total"][1:]) - np.array(loss["total"][:-1]))
        == -1
    )
