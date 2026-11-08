"""Guard the NLPOptNet dependency direction."""

import ast
from pathlib import Path
import sys


def main(root: Path) -> None:
    forbidden = {"kkthn", "daehn", "jaxmodel", "opt"}
    violations = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.partition(".")[0] in forbidden:
                    violations.append(f"{path}:{node.lineno}: {name}")
    if violations:
        raise SystemExit("\n".join(violations))
    print("Frontend dependency boundaries passed.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
