# Contributing

## Building the docs

We recommend that you use the latest version
of python when contributing.
Empirically, while all tests run for a wide range of python versions,
we know that building the docs with python 3.7 can raise
errors which do not occur while building it with python 3.11.

```bash
pip install -e ".[dev]"
```

```bash
cd doc
make clean && make html
```
