import torch
import pytest
import warnings


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "skip_if_no_mkl: Skip test if MKL support is not available."
    )


@pytest.fixture(autouse=True)
def check_mkl_availability(request):
    if (
        "skip_if_no_mkl" in request.keywords
        and not torch.backends.mkl.is_available()
    ):
        pytest.skip("Test requires MKL support which is not available.")


@pytest.fixture(scope="session", autouse=True)
def ignore_sparse_csr_warning():
    """Remove the warning for sparse CSR tensor support."""
    warnings.filterwarnings(
        "ignore", ".*Sparse CSR tensor support is in beta state.*"
    )
