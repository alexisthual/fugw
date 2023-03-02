# Fused Unbalanced Gromov-Wasserstein for Python

![build](https://img.shields.io/github/actions/workflow/status/alexisthual/fugw/unit_tests.yml?event=push&style=for-the-badge)
![python version](https://img.shields.io/badge/python-3.7_|_3.8_|_3.9_|_3.10_|_3.11-blue?style=for-the-badge)
![license](https://img.shields.io/github/license/alexisthual/fugw?style=for-the-badge)
![code style](https://img.shields.io/badge/code_style-black-black?style=for-the-badge)

This package implements multiple GPU-compatible PyTorch solvers
to the Fused Unbalanced Gromov-Wasserstein optimal transport problem.

**This package is under active development. There is no guarantee that the API and solvers
won't change in the near future.**

## Installation

**To install this package, make sure you have an up-to-date version of** `pip`.

### From PyPI

In a dedicated Python env, run:

```bash
pip install fugw
```

### From source

```bash
git clone https://github.com/alexisthual/fugw.git
cd fugw
```

In a dedicated Python env, run:

```bash
pip install -e .
```

Contributors should also install the development dependencies
in order to test and automatically format their contributions.

```bash
pip install -e ".[dev]"
pre-commit install
```

Tests run on CPU and GPU, depending on the configuration of your machine.
You can run them with:

```bash
pytest
```

## Citing this work

If this package was useful to you, please cite it in your work:

```bibtex
@article{Thual-2022-fugw,
  title={Aligning individual brains with Fused Unbalanced Gromov-Wasserstein},
  author={Thual, Alexis and Tran, Huy and Zemskova, Tatiana and Courty, Nicolas and Flamary, Rémi and Dehaene, Stanislas and Thirion, Bertrand},
  publisher={arXiv},
  doi={10.48550/ARXIV.2206.09398},
  url={https://arxiv.org/abs/2206.09398},
  year={2022},
  copyright={Creative Commons Attribution 4.0 International}
}
```
