# Installation

These steps assume you cloned the repository from GitHub and are inside the
`NLPOptNet` directory.

## Windows PowerShell

```powershell
python -m venv D:\Projects\virtual_envs\env
D:\Projects\virtual_envs\env\Scripts\Activate.ps1

python -m pip install --upgrade pip
python nlpopt/install_info.py
pip install -e nlpopt
```

If PowerShell blocks activation scripts, use this for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
D:\Projects\virtual_envs\env\Scripts\Activate.ps1
```

## macOS, Linux, WSL

```bash
python3 -m venv env
source env/bin/activate

python -m pip install --upgrade pip
python nlpopt/install_info.py
pip install -e nlpopt
```

## CPU/GPU Selection

`pip install -e nlpopt` selects dependencies automatically:

- CPU requirements on native Windows or when CUDA is not detected.
- GPU requirements when CUDA is detected on Linux/WSL.

To force one mode:

```bash
NLPOPT_REQUIREMENTS=cpu pip install -e nlpopt
NLPOPT_REQUIREMENTS=gpu pip install -e nlpopt
```

On Windows PowerShell:

```powershell
$env:NLPOPT_REQUIREMENTS = "cpu"
pip install -e nlpopt
```
