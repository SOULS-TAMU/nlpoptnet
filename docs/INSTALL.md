# Installation

These instructions assume you are in the repository root.

## Local Editable Install

```bash
python -m pip install --upgrade pip
pip install -e nlpoptnet
```

That installs the package from:

```text
nlpoptnet/
```

## Published Install

After publishing to PyPI:

```bash
pip install nlpoptnet
```

## Python Support

The package targets:

- Python `>=3.9`
- Python `3.13` with dependency markers that select newer `jax`, `jaxlib`,
  `numpy`, and `scipy` wheels

The package metadata in [`nlpoptnet/pyproject.toml`](../nlpoptnet/pyproject.toml)
uses environment markers so `pip` installs a compatible dependency set for
`python_version < "3.13"` versus `python_version >= "3.13"`.

## Fresh Environment

macOS, Linux, or WSL:

```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip install -e nlpoptnet
```

Windows PowerShell:

```powershell
python -m venv env
env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e nlpoptnet
```

If PowerShell blocks activation scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
env\Scripts\Activate.ps1
```

## Development Extras

For packaging and release work:

```bash
python -m pip install --upgrade build twine
```
