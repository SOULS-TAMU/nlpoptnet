# nlpopt

Install the NLPOptNet package from the repository root:

```bash
pip install -e nlpopt
```

The editable install selects package-local CPU or GPU dependencies at install
time. Use `python nlpopt/install_info.py` to see which dependency set will be
used before installation.

Public imports:

```python
from nlpopt import ProblemBuilder
from jaxmodel import HighLevelNLPBuilder
from opt.training import TrainConfig
```
