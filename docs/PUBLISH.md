# Publishing

The installable package is rooted at:

```text
nlpoptnet/
```

## 1. Bump the Version

Update both:

- `nlpoptnet/pyproject.toml`
- `nlpoptnet/src/nlpoptnet/__init__.py`

## 2. Build the Package

From the repository root:

```bash
python -m pip install --upgrade build twine
cd nlpoptnet
python -m build
```

This creates:

```text
nlpoptnet/dist/
```

## 3. Check the Distribution

```bash
python -m twine check dist/*
```

## 4. Upload

TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

PyPI:

```bash
python -m twine upload dist/*
```

## 5. Verify

In a clean environment:

```bash
pip install nlpoptnet
python -c "from nlpoptnet import NLPOptNet; print(NLPOptNet)"
```
