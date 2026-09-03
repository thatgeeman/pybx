## Setup development environment

PyBx supports Python 3.8 through 3.14. The nbdev 3 documentation tools
require Python 3.10 or newer, so use a recent interpreter for development:

```shell
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the test suite with `python -m pytest -q` and the nbdev checks with
`nbdev-prepare`.

## Build

```shell
python -m build
```

## Editable install
```bash
python -m pip install -e .
```
