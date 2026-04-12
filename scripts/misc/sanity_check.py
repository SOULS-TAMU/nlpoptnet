#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    src = root / "nlpopt" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import jax  # noqa: F401
    import cvxpy  # noqa: F401
    import jaxmodel  # noqa: F401
    import opt  # noqa: F401
    import solgen  # noqa: F401

    required = [root / "data.json", root / "config.json", root / "proj.json"]
    for path in required:
        with open(path, "r") as fh:
            json.load(fh)

    print("NLPOpt sanity check passed.")
    print(f"Project root: {root}")
    print(f"Source root: {src}")
    print("Configs present: data.json, config.json, proj.json")
    print("Imports present: jaxmodel, opt, solgen, cvxpy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
