# Contributing

## Installing development dependencies

```bash
pip install -e ".[dev]"
```

## Formatting committed code

Commits must validate a pre-commit routine launched by a git hook.
To enable this hook locally, run

```bash
pre-commit install
```

## Building the docs

```bash
cd doc
make clean && make html
```
