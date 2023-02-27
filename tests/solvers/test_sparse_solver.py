from itertools import product

import numpy as np
import pytest
import torch

from fugw.solvers import FUGWSparseSolver
from fugw.utils import low_rank_squared_l2


devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))


# TODO: need to test sinkhorn
@pytest.mark.parametrize(
    "uot_solver,device",
    product(["mm", "ibpp"], devices),
)
def test_sparse_solvers(uot_solver, device):
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

    F = low_rank_squared_l2(source_features, target_features)
    Ds = low_rank_squared_l2(source_embeddings, source_embeddings)
    Dt = low_rank_squared_l2(target_embeddings, target_embeddings)

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
        uot_solver=uot_solver,
        verbose=True,
    )

    pi = res["pi"]
    gamma = res["gamma"]
    duals_pi = res["duals_pi"]
    duals_gamma = res["duals_gamma"]
    loss_steps = res["loss_steps"]
    loss = res["loss"]
    loss_entropic = res["loss_entropic"]
    loss_times = res["loss_times"]

    assert pi.size() == (ns, nt)
    assert gamma.size() == (ns, nt)

    if uot_solver == "mm":
        assert duals_pi is None
        assert duals_gamma is None
    elif uot_solver == "ibpp":
        assert len(duals_pi) == 2
        assert duals_pi[0].shape == (ns,)
        assert duals_pi[1].shape == (nt,)

    assert len(loss_steps) - 1 <= nits_bcd // eval_bcd + 1
    assert len(loss) == len(loss_steps)
    assert len(loss_entropic) == len(loss_steps)
    assert len(loss_times) == len(loss_steps)
    # Loss should decrease
    print(f"loss: {loss}")
    assert np.all(np.sign(np.array(loss[1:]) - np.array(loss[:-1])) == -1)
