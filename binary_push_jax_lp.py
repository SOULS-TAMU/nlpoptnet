"""
Sequential convex relaxation with binary push for random mixed-binary LPs.

Problem form
------------
    min_x      c^T x
    s.t.       A_eq x = b_eq
               A_ineq x <= b_ineq
               0 <= x_i <= 1                  for continuous variables
               x_j in {0,1}                  for binary variables in the MIP
               0 <= x_j <= 1                 for binary variables in LP relaxations

The binary-push logic mirrors the toy MATLAB implementation LP_04.m:
  1. solve an LP relaxation;
  2. compute objective and active-constraint contributions for fractional binary coordinates;
  3. compute beta_star and choose beta slightly to one side of the switching surface;
  4. optionally force the direction away from infeasible binary endpoints;
  5. add the corrected linearized push coefficient
         r_j = -2 * gamma_j * M_j * abs(z_j^* - 1/2),
     where M_j = -1 means push to 0 and M_j = +1 means push to 1;
  6. solve the modified LP and iterate.

Dependencies
------------
Required: numpy, scipy, jax, jaxlib
Optional: gurobipy, for exact mixed-binary LP comparison.

Example
-------
    python binary_push_jax_lp.py --n_cont 8 --n_bin 12 --n_eq 3 --n_ineq 30 --seed 7
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Optional, Tuple, Dict, Any, List
import argparse
import math

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import linprog


@dataclass
class RandomMILP:
    c: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    A_ineq: np.ndarray
    b_ineq: np.ndarray
    n_cont: int
    n_bin: int
    x_feas: np.ndarray

    @property
    def n(self) -> int:
        return self.n_cont + self.n_bin

    @property
    def bin_idx(self) -> np.ndarray:
        return np.arange(self.n_cont, self.n_cont + self.n_bin)

    @property
    def cont_idx(self) -> np.ndarray:
        return np.arange(self.n_cont)


def _as_np(x: jnp.ndarray) -> np.ndarray:
    return np.asarray(jax.device_get(x), dtype=float)


def generate_random_lp(
    n_cont: int,
    n_bin: int,
    n_eq: int,
    n_ineq: int,
    seed: int = 0,
    slack_scale: float = 0.75,
    objective_scale: float = 1.0,
    eq_rank: Optional[int] = None,
    min_cont_value: float = 0.05,
    max_cont_value: float = 0.95,
    validate: bool = True,
) -> RandomMILP:
    """Generate a bounded feasible random mixed-binary LP.

    A planted feasible point is created first. Equality RHS values are generated
    from that point, and inequality RHS values include nonnegative slack.
    All arrays are built in NumPy float64 so the planted point satisfies the
    generated equality system to solver tolerances even when the number of
    equalities is large. Requested equality rows are generated with a controlled
    effective rank so a large ``n_eq`` does not accidentally pin every variable
    to the planted point. All variables are bounded in [0, 1]. Binary variables
    are integral only in the exact benchmark; the relaxation uses [0, 1].
    """
    n = n_cont + n_bin
    if n <= 0:
        raise ValueError("n_cont + n_bin must be positive.")
    if n_cont < 0 or n_bin < 0 or n_eq < 0 or n_ineq < 0:
        raise ValueError("Problem dimensions must be nonnegative.")
    if not (0.0 <= min_cont_value <= max_cont_value <= 1.0):
        raise ValueError("Continuous planted-value bounds must satisfy 0 <= min <= max <= 1.")
    if slack_scale <= 0:
        raise ValueError("slack_scale must be positive.")

    rng = np.random.default_rng(seed)
    scale = math.sqrt(max(n, 1))

    x_cont = rng.uniform(min_cont_value, max_cont_value, size=n_cont)
    z_bin = rng.integers(0, 2, size=n_bin).astype(np.float64)
    x_feas = np.concatenate([x_cont, z_bin]).astype(np.float64)

    effective_eq_rank = _effective_equality_rank(n_cont, n_bin, n_eq, eq_rank)
    A_eq = _generate_rank_controlled_matrix(rng, n_eq, n, effective_eq_rank, scale)
    b_eq = A_eq @ x_feas

    A_ineq = rng.normal(size=(n_ineq, n)).astype(np.float64) / scale if n_ineq else np.zeros((0, n), dtype=np.float64)
    slack = rng.exponential(scale=slack_scale, size=n_ineq).astype(np.float64) if n_ineq else np.zeros((0,), dtype=np.float64)
    b_ineq = A_ineq @ x_feas + slack

    # Include a mild bias on binary costs so the true MIP has meaningful binary decisions.
    c_cont = objective_scale * rng.normal(size=n_cont)
    c_bin = objective_scale * rng.normal(size=n_bin)
    c = np.concatenate([c_cont, c_bin]).astype(np.float64)

    problem = RandomMILP(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        n_cont=n_cont,
        n_bin=n_bin,
        x_feas=x_feas,
    )
    if validate:
        residual = feasibility_residual(problem, problem.x_feas)
        if residual["max"] > 1e-9:
            raise RuntimeError(f"Generated planted point is not feasible enough: {residual}")
    return problem


def _effective_equality_rank(n_cont: int, n_bin: int, n_eq: int, eq_rank: Optional[int]) -> int:
    n = n_cont + n_bin
    if n_eq == 0:
        return 0
    max_rank = min(n_eq, n)
    if eq_rank is not None:
        if eq_rank < 0 or eq_rank > max_rank:
            raise ValueError(f"eq_rank must be between 0 and {max_rank}.")
        return int(eq_rank)

    if n <= 1:
        return max_rank
    if n_cont > 0:
        default_rank = min(n_cont, n - 1)
    else:
        default_rank = max(1, n // 2)
    return min(max_rank, default_rank)


def _generate_rank_controlled_matrix(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    rank: int,
    scale: float,
) -> np.ndarray:
    if rows == 0:
        return np.zeros((0, cols), dtype=np.float64)
    if rank == 0:
        return np.zeros((rows, cols), dtype=np.float64)
    if rank == rows:
        A = rng.normal(size=(rows, cols)).astype(np.float64) / scale
    else:
        basis = rng.normal(size=(rank, cols)).astype(np.float64) / scale
        mix = rng.normal(size=(rows, rank)).astype(np.float64)
        mix[:rank, :] = np.eye(rank, dtype=np.float64)
        A = mix @ basis

    row_norm = np.linalg.norm(A, axis=1)
    nonzero = row_norm > 0.0
    A[nonzero] = A[nonzero] / row_norm[nonzero, None]
    return A


def solve_lp_relaxation(problem: RandomMILP, c: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Solve the LP relaxation with SciPy HiGHS and return primal and dual data."""
    if c is None:
        c = problem.c
    bounds = [(0.0, 1.0)] * problem.n
    res = linprog(
        c,
        A_ub=problem.A_ineq if problem.A_ineq.shape[0] else None,
        b_ub=problem.b_ineq if problem.b_ineq.shape[0] else None,
        A_eq=problem.A_eq if problem.A_eq.shape[0] else None,
        b_eq=problem.b_eq if problem.b_eq.shape[0] else None,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP solve failed: {res.message}")

    # HiGHS reports RHS marginals. For A_ub x <= b_ub in minimization, these
    # marginals usually have the opposite sign of nonnegative Lagrange multipliers.
    # We convert to nonnegative-ish mu for the uploaded-theory convention.
    ineqlin_marginals = np.asarray(res.ineqlin.marginals, dtype=float) if problem.A_ineq.shape[0] else np.zeros(0)
    eqlin_marginals = np.asarray(res.eqlin.marginals, dtype=float) if problem.A_eq.shape[0] else np.zeros(0)
    lower_marginals = np.asarray(res.lower.marginals, dtype=float)
    upper_marginals = np.asarray(res.upper.marginals, dtype=float)

    mu_ineq = -ineqlin_marginals
    lam_eq = eqlin_marginals

    return {
        "x": np.asarray(res.x, dtype=float),
        "objective": float(res.fun),
        "result": res,
        "mu_ineq": mu_ineq,
        "lambda_eq": lam_eq,
        "lower_marginals": lower_marginals,
        "upper_marginals": upper_marginals,
    }


def feasibility_residual(problem: RandomMILP, x: Optional[np.ndarray]) -> Dict[str, float]:
    """Return feasibility and integrality residuals for a candidate solution."""
    if x is None:
        return {
            "eq_inf": np.inf,
            "ineq_inf": np.inf,
            "bounds_inf": np.inf,
            "binary_inf": np.inf,
            "max": np.inf,
        }

    residual = {"eq_inf": 0.0, "ineq_inf": 0.0, "bounds_inf": 0.0, "binary_inf": 0.0}
    if problem.A_eq.shape[0]:
        residual["eq_inf"] = float(np.max(np.abs(problem.A_eq @ x - problem.b_eq)))
    if problem.A_ineq.shape[0]:
        residual["ineq_inf"] = float(max(0.0, np.max(problem.A_ineq @ x - problem.b_ineq)))
    residual["bounds_inf"] = float(max(0.0, np.max(-x), np.max(x - 1.0)))
    residual["binary_inf"] = float(np.max(np.abs(x[problem.bin_idx] - np.round(x[problem.bin_idx]))))
    residual["max"] = max(residual.values())
    return residual


def endpoint_feasible(problem: RandomMILP, j: int, value: float, tol: float = 1e-8) -> bool:
    """Check whether the LP relaxation is feasible after fixing x_j=value."""
    bounds = [(0.0, 1.0)] * problem.n
    bounds[j] = (float(value), float(value))
    res = linprog(
        np.zeros(problem.n),
        A_ub=problem.A_ineq if problem.A_ineq.shape[0] else None,
        b_ub=problem.b_ineq if problem.A_ineq.shape[0] else None,
        A_eq=problem.A_eq if problem.A_eq.shape[0] else None,
        b_eq=problem.b_eq if problem.A_eq.shape[0] else None,
        bounds=bounds,
        method="highs",
    )
    return bool(res.success)


def compute_binary_push(
    problem: RandomMILP,
    lp: Dict[str, Any],
    beta_offset_fraction: float = 0.01,
    use_endpoint_forcing: bool = True,
    frac_tol: float = 1e-6,
    score_tol: float = 1e-10,
) -> Dict[str, np.ndarray]:
    """Compute beta, direction M, and unit-gamma shift rho for each binary variable."""
    x = lp["x"]
    bin_idx = problem.bin_idx
    n_bin = problem.n_bin

    objective_part = problem.c[bin_idx].copy()

    # Non-bound contribution: equality + inequality contributions.
    # Equality multipliers are unrestricted. Inequality multipliers are converted
    # to the nonnegative convention used by the theory / MATLAB prototype.
    constraint_part = np.zeros(n_bin)
    if problem.A_ineq.shape[0]:
        constraint_part += problem.A_ineq[:, bin_idx].T @ lp["mu_ineq"]
    if problem.A_eq.shape[0]:
        constraint_part += problem.A_eq[:, bin_idx].T @ lp["lambda_eq"]

    denom = objective_part - constraint_part
    beta_star = np.empty(n_bin)
    mask_small = np.abs(denom) < 1e-12
    beta_star[mask_small] = 0.5
    beta_star[~mask_small] = -constraint_part[~mask_small] / denom[~mask_small]
    beta_star = np.clip(beta_star, 0.0, 1.0)

    # MATLAB-style offset: beta = beta_star + beta_star / 100.
    # If beta_star is zero, step slightly inside the interval to break ties.
    beta = beta_star + beta_offset_fraction * np.maximum(beta_star, 1.0)
    beta = np.clip(beta, 0.0, 1.0)

    score = beta * objective_part + (1.0 - beta) * constraint_part
    procedure_direction = np.sign(score)
    procedure_direction[np.abs(score) < score_tol] = 1.0

    final_direction = procedure_direction.copy()
    has_zero_endpoint = np.ones(n_bin, dtype=bool)
    has_one_endpoint = np.ones(n_bin, dtype=bool)

    if use_endpoint_forcing:
        for k, j in enumerate(bin_idx):
            z = x[j]
            if frac_tol < z < 1.0 - frac_tol:
                has_zero_endpoint[k] = endpoint_feasible(problem, int(j), 0.0)
                has_one_endpoint[k] = endpoint_feasible(problem, int(j), 1.0)
                if has_zero_endpoint[k] and has_one_endpoint[k]:
                    final_direction[k] = procedure_direction[k]
                elif not has_one_endpoint[k] and has_zero_endpoint[k]:
                    final_direction[k] = -1.0
                elif not has_zero_endpoint[k] and has_one_endpoint[k]:
                    final_direction[k] = 1.0
                # If neither endpoint is feasible, keep procedure direction.

    z = x[bin_idx]
    fractional = (z > frac_tol) & (z < 1.0 - frac_tol)

    # rho is the coefficient shift per unit gamma using corrected orientation.
    rho_bin = np.zeros(n_bin)
    rho_bin[fractional] = -2.0 * final_direction[fractional] * np.abs(z[fractional] - 0.5)

    return {
        "objective_part": objective_part,
        "constraint_part": constraint_part,
        "beta_star": beta_star,
        "beta": beta,
        "score": score,
        "procedure_direction": procedure_direction,
        "direction": final_direction,
        "rho_bin": rho_bin,
        "fractional": fractional.astype(bool),
        "has_zero_endpoint": has_zero_endpoint,
        "has_one_endpoint": has_one_endpoint,
    }


def sequential_binary_push(
    problem: RandomMILP,
    max_iter: int = 25,
    gamma0: float = 1.0,
    gamma_growth: float = 1.5,
    beta_offset_fraction: float = 0.01,
    frac_tol: float = 1e-6,
    round_final: bool = True,
    repair_flip_depth: int = 2,
) -> Dict[str, Any]:
    """Run sequential LP relaxations with binary-push coefficient updates.

    Every convex reformulation can yield a different rounded/fixed-binary
    feasible point. The returned candidate is the best repaired point observed
    across the whole reformulation history, measured with the original LP
    objective. This avoids returning a late iterate from a cycling sequence when
    an earlier reformulation already found a better integer-feasible point.
    """
    c_mod = problem.c.copy()
    gamma = float(gamma0)
    history: List[Dict[str, Any]] = []
    candidate_history: List[Dict[str, Any]] = []
    best_x: Optional[np.ndarray] = None
    best_objective = np.inf
    best_iteration: Optional[int] = None

    for it in range(max_iter):
        lp = solve_lp_relaxation(problem, c=c_mod)
        push = compute_binary_push(
            problem,
            lp,
            beta_offset_fraction=beta_offset_fraction,
            use_endpoint_forcing=True,
            frac_tol=frac_tol,
        )
        x = lp["x"]
        z = x[problem.bin_idx]
        n_frac = int(np.sum((z > frac_tol) & (z < 1.0 - frac_tol)))

        true_obj = float(problem.c @ x)
        history.append({
            "iter": it,
            "gamma": gamma,
            "lp_modified_objective": lp["objective"],
            "original_objective_at_lp_solution": true_obj,
            "n_fractional": n_frac,
            "x": x.copy(),
            "push": push,
        })

        repaired = (
            repair_by_fixing_binaries(problem, x, threshold=0.5, local_search_depth=repair_flip_depth)
            if round_final
            else x.copy()
        )
        if repaired is not None:
            repaired_objective = float(problem.c @ repaired)
            repaired_residual = feasibility_residual(problem, repaired)
            candidate_history.append({
                "iter": it,
                "objective": repaired_objective,
                "x": repaired.copy(),
                "residual": repaired_residual,
            })
            if repaired_objective < best_objective:
                best_objective = repaired_objective
                best_x = repaired.copy()
                best_iteration = it

        if n_frac == 0:
            break

        rho = np.zeros(problem.n)
        rho[problem.bin_idx] = push["rho_bin"]
        c_mod = problem.c + gamma * rho
        gamma *= gamma_growth

    final_x_relax = history[-1]["x"].copy()

    if round_final:
        final_x = (
            best_x
            if best_x is not None
            else repair_by_fixing_binaries(problem, final_x_relax, threshold=0.5, local_search_depth=repair_flip_depth)
        )
    else:
        final_x = final_x_relax

    return {
        "x_relax_final": final_x_relax,
        "x_candidate": final_x,
        "objective_candidate": float(problem.c @ final_x) if final_x is not None else np.inf,
        "feasibility": feasibility_residual(problem, final_x),
        "best_iteration": best_iteration,
        "history": history,
        "candidate_history": candidate_history,
    }


def repair_by_fixing_binaries(
    problem: RandomMILP,
    x_relax: np.ndarray,
    threshold: float = 0.5,
    local_search_depth: int = 2,
) -> Optional[np.ndarray]:
    """Round binaries, fix them, and re-optimize continuous variables by LP.

    The initial pool includes the rounded binary pattern plus nearby one- and
    two-bit flips around the most fractional coordinates. The best feasible
    member of that pool is then improved by a small fixed-binary local search.
    """
    bin_idx = problem.bin_idx
    z_relax = x_relax[bin_idx]
    z0 = (z_relax >= threshold).astype(float)
    order = np.argsort(np.abs(z_relax - 0.5))

    candidates = [z0]
    # Single flips, then double flips for a modest repair attempt.
    for k in order:
        z = z0.copy()
        z[k] = 1.0 - z[k]
        candidates.append(z)
    max_double = min(len(order), 10)
    for a in range(max_double):
        for b in range(a + 1, max_double):
            z = z0.copy()
            z[order[a]] = 1.0 - z[order[a]]
            z[order[b]] = 1.0 - z[order[b]]
            candidates.append(z)

    best_x: Optional[np.ndarray] = None
    best_z: Optional[np.ndarray] = None
    best_objective = np.inf
    for z in candidates:
        x_fixed = solve_fixed_binary_pattern(problem, z)
        if x_fixed is None:
            continue
        objective = float(problem.c @ x_fixed)
        if objective < best_objective:
            best_objective = objective
            best_x = x_fixed
            best_z = z.copy()

    if best_x is None or best_z is None:
        return None

    if local_search_depth <= 0:
        return best_x

    move_sizes = range(1, min(int(local_search_depth), problem.n_bin) + 1)
    moves = [move for size in move_sizes for move in combinations(range(problem.n_bin), size)]
    improved = True
    while improved:
        improved = False
        for move in moves:
            z = best_z.copy()
            for local_k in move:
                z[local_k] = 1.0 - z[local_k]
            x_fixed = solve_fixed_binary_pattern(problem, z)
            if x_fixed is None:
                continue
            objective = float(problem.c @ x_fixed)
            if objective < best_objective - 1e-9:
                best_objective = objective
                best_x = x_fixed
                best_z = z
                improved = True
                break
    return best_x


def solve_fixed_binary_pattern(problem: RandomMILP, z_binary: np.ndarray) -> Optional[np.ndarray]:
    """Solve the continuous LP left after fixing all binary variables."""
    bounds = [(0.0, 1.0)] * problem.n
    for local_k, j in enumerate(problem.bin_idx):
        bounds[int(j)] = (float(z_binary[local_k]), float(z_binary[local_k]))
    res = linprog(
        problem.c,
        A_ub=problem.A_ineq if problem.A_ineq.shape[0] else None,
        b_ub=problem.b_ineq if problem.A_ineq.shape[0] else None,
        A_eq=problem.A_eq if problem.A_eq.shape[0] else None,
        b_eq=problem.b_eq if problem.A_eq.shape[0] else None,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return None
    return np.asarray(res.x, dtype=float)


def solve_with_gurobi(problem: RandomMILP, time_limit: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Solve the original mixed-binary LP with Gurobi, if available."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        return {"available": False, "reason": f"gurobipy import failed: {exc}"}

    try:
        model = gp.Model("random_binary_push_lp")
        model.Params.OutputFlag = 0
        if time_limit is not None:
            model.Params.TimeLimit = time_limit

        x = []
        for i in range(problem.n_cont):
            x.append(model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"u_{i}"))
        for j in range(problem.n_bin):
            x.append(model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"z_{j}"))
        model.update()

        model.setObjective(gp.quicksum(float(problem.c[i]) * x[i] for i in range(problem.n)), GRB.MINIMIZE)

        for r in range(problem.A_eq.shape[0]):
            model.addConstr(gp.quicksum(float(problem.A_eq[r, i]) * x[i] for i in range(problem.n)) == float(problem.b_eq[r]))
        for r in range(problem.A_ineq.shape[0]):
            model.addConstr(gp.quicksum(float(problem.A_ineq[r, i]) * x[i] for i in range(problem.n)) <= float(problem.b_ineq[r]))

        model.optimize()
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            return {"available": True, "status": int(model.Status), "success": False}

        sol = np.array([var.X for var in x], dtype=float)
        return {
            "available": True,
            "success": True,
            "status": int(model.Status),
            "objective": float(model.ObjVal),
            "x": sol,
            "mip_gap": float(model.MIPGap) if hasattr(model, "MIPGap") else None,
        }
    except Exception as exc:
        return {"available": False, "reason": f"Gurobi solve failed: {exc}"}


def solve_exact_by_enumeration(problem: RandomMILP, max_binary: int = 16) -> Dict[str, Any]:
    """Solve a small mixed-binary LP exactly by enumerating binary assignments."""
    if problem.n_bin > max_binary:
        return {
            "available": False,
            "success": False,
            "reason": f"{problem.n_bin} binaries exceed exact enumeration limit {max_binary}",
        }

    best_x: Optional[np.ndarray] = None
    best_objective = np.inf
    feasible_assignments = 0

    for bits in product((0.0, 1.0), repeat=problem.n_bin):
        bounds = [(0.0, 1.0)] * problem.n
        for k, j in enumerate(problem.bin_idx):
            bounds[int(j)] = (bits[k], bits[k])
        res = linprog(
            problem.c,
            A_ub=problem.A_ineq if problem.A_ineq.shape[0] else None,
            b_ub=problem.b_ineq if problem.A_ineq.shape[0] else None,
            A_eq=problem.A_eq if problem.A_eq.shape[0] else None,
            b_eq=problem.b_eq if problem.A_eq.shape[0] else None,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            continue
        feasible_assignments += 1
        if float(res.fun) < best_objective:
            best_objective = float(res.fun)
            best_x = np.asarray(res.x, dtype=float)

    if best_x is None:
        return {
            "available": True,
            "success": False,
            "reason": "No feasible binary assignment found.",
            "feasible_assignments": feasible_assignments,
        }

    return {
        "available": True,
        "success": True,
        "status": 2,
        "backend": "scipy-enumeration",
        "objective": best_objective,
        "x": best_x,
        "feasible_assignments": feasible_assignments,
        "residual": feasibility_residual(problem, best_x),
    }


def compare_solutions(problem: RandomMILP, bp: Dict[str, Any], benchmark: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    x_bp = bp["x_candidate"]
    out = {
        "binary_push_objective": bp["objective_candidate"],
        "binary_push_x": x_bp,
        "binary_push_feasibility": bp["feasibility"],
        "binary_push_best_iteration": bp["best_iteration"],
    }
    if benchmark and benchmark.get("available") and benchmark.get("success"):
        x_g = benchmark["x"]
        out.update({
            "benchmark_backend": benchmark.get("backend", "gurobi"),
            "benchmark_objective": benchmark["objective"],
            "objective_gap_abs": float(bp["objective_candidate"] - benchmark["objective"]),
            "objective_gap_rel": float((bp["objective_candidate"] - benchmark["objective"]) / max(1.0, abs(benchmark["objective"]))),
            "variable_linf_distance": float(np.max(np.abs(x_bp - x_g))) if x_bp is not None else np.inf,
            "binary_hamming_distance": int(np.sum(np.round(x_bp[problem.bin_idx]) != np.round(x_g[problem.bin_idx]))) if x_bp is not None else problem.n_bin,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cont", type=int, default=8)
    parser.add_argument("--n_bin", type=int, default=12)
    parser.add_argument("--n_eq", type=int, default=3)
    parser.add_argument("--n_ineq", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--eq_rank",
        type=int,
        default=None,
        help="Effective rank of generated equality rows. Defaults to a rank that leaves relaxation degrees of freedom.",
    )
    parser.add_argument("--max_iter", type=int, default=25)
    parser.add_argument("--gamma0", type=float, default=1.0)
    parser.add_argument("--gamma_growth", type=float, default=1.5)
    parser.add_argument("--gurobi_time_limit", type=float, default=None)
    parser.add_argument("--exact_enum_max_binary", type=int, default=16)
    parser.add_argument("--repair_flip_depth", type=int, default=2)
    parser.add_argument(
        "--convex_reformulations",
        type=int,
        default=None,
        help="Alias for --max_iter; number of sequential convex LP reformulations.",
    )
    args = parser.parse_args()
    if args.convex_reformulations is not None:
        args.max_iter = args.convex_reformulations

    problem = generate_random_lp(
        n_cont=args.n_cont,
        n_bin=args.n_bin,
        n_eq=args.n_eq,
        n_ineq=args.n_ineq,
        seed=args.seed,
        eq_rank=args.eq_rank,
    )

    bp = sequential_binary_push(
        problem,
        max_iter=args.max_iter,
        gamma0=args.gamma0,
        gamma_growth=args.gamma_growth,
        repair_flip_depth=args.repair_flip_depth,
    )
    grb = solve_with_gurobi(problem, time_limit=args.gurobi_time_limit)
    benchmark = grb
    if not (grb and grb.get("available") and grb.get("success")):
        enum = solve_exact_by_enumeration(problem, max_binary=args.exact_enum_max_binary)
        benchmark = enum
    cmp = compare_solutions(problem, bp, benchmark)

    print("\n=== Random mixed-binary LP ===")
    print(f"n_cont={args.n_cont}, n_bin={args.n_bin}, n_eq={args.n_eq}, n_ineq={args.n_ineq}, seed={args.seed}")
    print("\n=== Sequential binary push history ===")
    for h in bp["history"]:
        print(
            f"iter={h['iter']:02d}  gamma={h['gamma']:.4g}  "
            f"n_frac={h['n_fractional']:02d}  "
            f"orig_obj_at_relax={h['original_objective_at_lp_solution']:.8g}"
        )

    print("\n=== Binary-push candidate ===")
    print(f"objective = {cmp['binary_push_objective']:.10g}")
    print(f"best reformulation iteration = {cmp['binary_push_best_iteration']}")
    print(f"max feasibility residual = {cmp['binary_push_feasibility']['max']:.4g}")
    print(f"x = {np.array2string(cmp['binary_push_x'], precision=4, suppress_small=True)}")

    print("\n=== Exact benchmark ===")
    if benchmark and benchmark.get("available") and benchmark.get("success"):
        print(f"backend   = {cmp['benchmark_backend']}")
        print(f"objective = {cmp['benchmark_objective']:.10g}")
        print(f"abs gap   = {cmp['objective_gap_abs']:.10g}")
        print(f"rel gap   = {cmp['objective_gap_rel']:.10g}")
        print(f"||x_bp - x_gurobi||_inf = {cmp['variable_linf_distance']:.4g}")
        print(f"binary hamming distance = {cmp['binary_hamming_distance']} / {problem.n_bin}")
        print(f"x = {np.array2string(benchmark['x'], precision=4, suppress_small=True)}")
    else:
        reason = benchmark.get("reason", "not available") if benchmark else "not available"
        grb_reason = grb.get("reason") if grb else None
        if grb_reason:
            print(f"Gurobi unavailable: {grb_reason}")
        print(f"Exact benchmark unavailable: {reason}")


if __name__ == "__main__":
    main()
