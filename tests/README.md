## Setup development environment
*WIP*

Make python3.7 environment:
```shell
pipenv --py 3.7
```
Start environment and update packages accoring to `Pipfile`.
```shell
pipenv shell 
pipenv update --dev
pipenv check
```

## Build

```shell
python -m build
```

## Editable install
```bash
pipenv run pip install -e . 
```