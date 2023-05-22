# %%
"""
Monitor memory usage at each BCD iteration with callbacks
=========================================================

In this example, we use a callback function to monitor memory usage
at each iteration of the block-coordinate descent (BCD) algorithm.
This can be useful to detect memory leaks, or to check that the
memory usage is not too high for a given device.
"""

import re

from functools import partial

import matplotlib.pyplot as plt
import torch

from fugw.mappings import FUGW
from fugw.utils import _init_mock_distribution
from rich.console import Console
from rich.table import Table

# %%
torch.manual_seed(0)

n_points_source = 50
n_points_target = 40
n_features_train = 2
n_features_test = 2

# %%
# Let us generate random training data for the source and target distributions
_, source_features_train, source_geometry, _ = _init_mock_distribution(
    n_features_train, n_points_source
)
_, target_features_train, target_geometry, _ = _init_mock_distribution(
    n_features_train, n_points_target
)

# %%
# Features and geometries should be normalized before calling the solver
source_features_train_normalized = source_features_train / torch.linalg.norm(
    source_features_train, dim=1
).reshape(-1, 1)
target_features_train_normalized = target_features_train / torch.linalg.norm(
    target_features_train, dim=1
).reshape(-1, 1)

source_geometry_normalized = source_geometry / source_geometry.max()
target_geometry_normalized = target_geometry / target_geometry.max()


# %%
# We define a function to check memory usage at each iteration of the BCD
# algorithm, as well as util functions to print relevant information.
# In short, fugw callback functions receive `locals()`, which is a dictionary
# of all local variables in the current scope.
# This allows us to access the tensors that are used in the BCD algorithm.
# In particular, we filter our tensors that are on our device of interest,
# and we compute their respective memory usage.
def is_sparse(t):
    return str(t.layout).find("sparse") >= 0


def str_size(s):
    m = re.match(r"torch\.Size\(\[(.*)\]\)", str(s))
    return f"{m.group(1)}"


def str_mem(mem, unit="KB"):
    if unit == "KB":
        return f"{mem / 1024:,.3f} KB"
    elif unit == "MB":
        return f"{mem / 1024 ** 2:,.3f} MB"


def check_memory_usage(locals, device=torch.device("cpu")):
    console = Console()

    variables = []
    for name, value in locals.items():
        if torch.is_tensor(value) and value.device == device:
            variables.append([name, value])

    variables = sorted(variables, key=lambda x: x[0].lower())

    table = Table()
    table.add_column("Variable")
    table.add_column("Size", justify="right")
    table.add_column("Numel", justify="right")
    table.add_column("Memory allocated", justify="right")

    memory_allocated = 0
    for name, value in variables:
        if is_sparse(value):
            s = value.size()
            numel = value._nnz()
            var_memory_allocated = numel * value.element_size()
            table.add_row(
                name, str_size(s), f"{numel:,}", str_mem(var_memory_allocated)
            )
        else:
            s = value.size()
            numel = value.numel()
            var_memory_allocated = numel * value.element_size()
            table.add_row(
                name, str_size(s), f"{numel:,}", str_mem(var_memory_allocated)
            )
        memory_allocated += var_memory_allocated

    table.add_section()
    table.add_row(
        f"Total ({len(variables)})",
        "",
        "",
        str_mem(memory_allocated),
        style="bold",
    )
    console.print(table)

    memory_at_bcd_step.append(memory_allocated)

    if device.type == "cuda":
        memory_lines = [
            ("Memory allocated", str_mem(torch.cuda.memory_allocated(device))),
            ("Memory cached", str_mem(torch.cuda.memory_cached(device))),
            ("Memory reserved", str_mem(torch.cuda.memory_reserved(device))),
        ]
        console.log(
            list(map(lambda x: f"{x[0]}\t{x[1]}", memory_lines)).join("\n")
        )


# %%
# Let us define the optimization problem to solve
alpha = 0.5
rho = 1000
eps = 1e-4
mapping = FUGW(alpha=alpha, rho=rho, eps=eps)

# %%
# Now, we fit a transport plan between source and target distributions
# using a sinkhorn solver.
# Our callback function will be called at each iteration of the BCD algorithm.
device = torch.device("cpu")
memory_at_bcd_step = []

_ = mapping.fit(
    source_features_train_normalized,
    target_features_train_normalized,
    source_geometry=source_geometry_normalized,
    target_geometry=target_geometry_normalized,
    solver="sinkhorn",
    solver_params={
        "nits_bcd": 5,
        "nits_uot": 100,
    },
    callback_bcd=partial(check_memory_usage, device=device),
    device=device,
    verbose=True,
)

# %%
# In this example, we see that fugw's memory usage is constant.
fig = plt.figure(figsize=(5, 5))
fig.suptitle("Memory usage at each BCD iteration")

ax = fig.add_subplot()
ax.set_xlabel("BCD iteration")
ax.set_xticks(range(len(memory_at_bcd_step)))
ax.set_ylabel("Memory allocated (KB)")
ax.plot(memory_at_bcd_step)

plt.show()
