from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linprog


def _can_parse_float_row(row: Sequence[str]) -> bool:
    try:
        [float(value) for value in row]
    except ValueError:
        return False
    return True


def load_csv_matrix(path: str | Path, expected_columns: list[str]) -> np.ndarray:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"CSV file not found: {target}")

    with open(target, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{target} is empty.")

    first = [cell.strip() for cell in rows[0]]
    expected = len(expected_columns)
    if expected <= 0:
        raise ValueError("expected_columns must be non-empty.")

    if len(first) == expected and _can_parse_float_row(first):
        data_rows = rows
    else:
        header = first
        missing = [name for name in expected_columns if name not in header]
        if missing:
            raise ValueError(f"{target} is missing columns: {missing}")
        index = [header.index(name) for name in expected_columns]
        data_rows = [[row[i] for i in index] for row in rows[1:] if any(cell.strip() for cell in row)]

    if not data_rows:
        raise ValueError(f"{target} has no data rows.")

    try:
        data = np.asarray([[float(cell) for cell in row] for row in data_rows], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{target} contains non-numeric data.") from exc

    if data.ndim != 2 or data.shape[1] != expected:
        raise ValueError(f"{target} must have exactly {expected} columns.")
    return data


def write_csv_matrix(path: str | Path, data: np.ndarray, headers: list[str] | None = None) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(data, dtype=np.float64)
    with open(target, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if headers is not None:
            writer.writerow(list(headers))
        writer.writerows(matrix.tolist())


def split_train_val(X: np.ndarray, *, train_frac: float, seed: int):
    X_np = np.asarray(X, dtype=np.float64)
    if X_np.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X_np.shape[0] < 2:
        raise ValueError("Need at least two parameter samples.")
    if not 0.0 < float(train_frac) < 1.0:
        raise ValueError("train_frac must satisfy 0 < train_frac < 1.")

    rng = np.random.default_rng(int(seed))
    indices = np.arange(int(X_np.shape[0]))
    rng.shuffle(indices)
    n_train = max(1, min(int(round(float(train_frac) * len(indices))), len(indices) - 1))
    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:])
    return train_idx, val_idx


def sample_box(lower: np.ndarray, upper: np.ndarray, *, num_samples: int, seed: int) -> np.ndarray:
    low = np.asarray(lower, dtype=np.float64).reshape(-1)
    high = np.asarray(upper, dtype=np.float64).reshape(-1)
    if low.shape != high.shape:
        raise ValueError("lower and upper must have the same shape.")
    if np.any(low > high):
        raise ValueError("Each lower bound must be <= the matching upper bound.")
    rng = np.random.default_rng(int(seed))
    return rng.uniform(low, high, size=(int(num_samples), int(low.size))).astype(np.float64)


def find_feasible_point(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A_np = np.asarray(A, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64).reshape(-1)
    if A_np.ndim != 2:
        raise ValueError("A must be a 2D array.")
    if A_np.shape[0] != b_np.shape[0]:
        raise ValueError("A and b shape mismatch.")
    result = linprog(
        c=np.zeros((A_np.shape[1],), dtype=np.float64),
        A_ub=A_np,
        b_ub=b_np,
        bounds=[(None, None)] * int(A_np.shape[1]),
        method="highs",
    )
    if not result.success or result.x is None:
        raise ValueError("Could not find a feasible point for the simplex/polytope constraints.")
    return np.asarray(result.x, dtype=np.float64)


def hit_and_run_samples(
    A: np.ndarray,
    b: np.ndarray,
    *,
    num_samples: int,
    seed: int,
    start: np.ndarray | None = None,
    burn_in: int | None = None,
    thinning: int | None = None,
) -> np.ndarray:
    A_np = np.asarray(A, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64).reshape(-1)
    if A_np.ndim != 2 or A_np.shape[0] != b_np.shape[0]:
        raise ValueError("A and b must define a valid halfspace system Ax <= b.")
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive.")

    dim = int(A_np.shape[1])
    x = find_feasible_point(A_np, b_np) if start is None else np.asarray(start, dtype=np.float64).reshape(dim)
    if np.any(A_np @ x - b_np > 1e-8):
        raise ValueError("The provided start point is not feasible.")

    rng = np.random.default_rng(int(seed))
    burn = int(burn_in) if burn_in is not None else max(25, 10 * dim)
    thin = int(thinning) if thinning is not None else max(3, dim)
    out = np.zeros((int(num_samples), dim), dtype=np.float64)

    total_steps = burn + thin * int(num_samples)
    saved = 0
    for step in range(total_steps):
        found_interval = False
        direction = None
        lower = None
        upper = None

        for _attempt in range(100):
            direction_candidate = rng.normal(size=(dim,))
            norm = float(np.linalg.norm(direction_candidate))
            if norm <= 1e-12:
                continue
            direction_candidate = direction_candidate / norm

            lower_candidate = -np.inf
            upper_candidate = np.inf
            for row, rhs in zip(A_np, b_np):
                denom = float(row @ direction_candidate)
                margin = float(rhs - row @ x)
                if abs(denom) <= 1e-12:
                    if margin < -1e-10:
                        raise ValueError("Current point left the feasible region during hit-and-run sampling.")
                    continue
                candidate = margin / denom
                if denom > 0.0:
                    upper_candidate = min(upper_candidate, candidate)
                else:
                    lower_candidate = max(lower_candidate, candidate)

            if lower_candidate > upper_candidate and lower_candidate <= upper_candidate + 1e-10:
                midpoint = 0.5 * (lower_candidate + upper_candidate)
                lower_candidate = midpoint
                upper_candidate = midpoint

            if np.isfinite(lower_candidate) and np.isfinite(upper_candidate) and lower_candidate <= upper_candidate:
                direction = direction_candidate
                lower = lower_candidate
                upper = upper_candidate
                found_interval = True
                break

        if not found_interval or direction is None or lower is None or upper is None:
            raise ValueError(
                "The simplex/polytope appears unbounded or numerically unstable for uniform sampling."
            )

        if abs(float(upper) - float(lower)) <= 1e-12:
            step_size = float(lower)
        else:
            lo = min(float(lower), float(upper))
            hi = max(float(lower), float(upper))
            step_size = float(rng.uniform(lo, hi))
        x = x + step_size * direction
        if step >= burn and (step - burn) % thin == 0:
            out[saved] = x
            saved += 1
            if saved >= int(num_samples):
                break

    if saved != int(num_samples):
        raise RuntimeError("Hit-and-run sampling terminated before collecting all requested samples.")
    return out
