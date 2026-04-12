from __future__ import annotations

from typing import Any


def fmt_sci(value: Any) -> str:
    return f"{float(value):.3e}"


def fmt_dec(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}"


def fmt_sec(value: Any) -> str:
    return f"{float(value):.3f}s"


def fmt_pct(value: Any) -> str:
    return f"{float(value):.3f}%"
