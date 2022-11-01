import pytest
import torch
from fugw.solvers.fugw import FUGWSolver


@pytest.mark.parametrize("uot_solver", ["sinkhorn", "mm", "dc"])
def test_solvers(uot_solver):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    nx = 104
    dx = 3
    ny = 151
    dy = 7

    x = torch.rand(nx, dx).to(device)
    y = torch.rand(ny, dy).to(device)

    # # if test with permutation
    # idx = torch.randperm(x.nelement())
    # y = x.view(-1)[idx].view(x.size())
    # ny = nx

    Cx = torch.cdist(x, x)
    Cy = torch.cdist(y, y)
    D = torch.rand(nx, ny)

    fugw = FUGWSolver(
        nits_bcd=100,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=1e-7,
        eval_bcd=2,
        eval_uot=10,
    )

    pi, gamma, duals_pi, duals_gamma, loss, loss_ent = fugw.solver(
        X=Cx,
        Y=Cy,
        D=D,
        alpha=0.8,
        rho_x=2,
        rho_y=3,
        eps=0.02,
        uot_solver=uot_solver,
        reg_mode="independent",
        init_plan=None,
        return_plans_only=False,
        verbose=True,
        early_stopping_threshold=1e-6,
    )

    assert pi.shape == (nx, ny)
    assert gamma.shape == (nx, ny)
