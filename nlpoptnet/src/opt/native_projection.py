"""Native sequential projection backend and compilation helpers."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


_NATIVE_FORMAT = "nlpoptnet-native-projection-v2"
_SOURCE_VERSION = "cp-sequential-20260422"
_C_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define NLPOPTNET_EXPORT __declspec(dllexport)
#else
#define NLPOPTNET_EXPORT
#endif

static double max2(double a, double b) { return a > b ? a : b; }
static double min2(double a, double b) { return a < b ? a : b; }
static double clip(double x, double lo, double hi) { return isnan(x) ? x : min2(max2(x, lo), hi); }

static void matvec(const double *M, int rows, int cols, const double *x, double *out) {
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int c = 0; c < cols; ++c) acc += M[(size_t)r * cols + c] * x[c];
        out[r] = acc;
    }
}

static void mat_t_vec_add(const double *M, int rows, int cols, const double *x, double *out) {
    for (int r = 0; r < rows; ++r) {
        const double xr = x[r];
        for (int c = 0; c < cols; ++c) out[c] += M[(size_t)r * cols + c] * xr;
    }
}

static double vec_norm(const double *x, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += x[i] * x[i];
    return sqrt(acc);
}

static double estimate_ac_norm(const double *A, int me, const double *C, int mi, int n, int iters, unsigned int seed) {
    if (n <= 0) return 1e-12;
    double *x = (double *)calloc((size_t)n, sizeof(double));
    double *y = (double *)calloc((size_t)n, sizeof(double));
    double *tmp_e = (double *)calloc((size_t)(me > 0 ? me : 1), sizeof(double));
    double *tmp_i = (double *)calloc((size_t)(mi > 0 ? mi : 1), sizeof(double));
    if (!x || !y || !tmp_e || !tmp_i) {
        free(x); free(y); free(tmp_e); free(tmp_i);
        return 1.0;
    }
    for (int j = 0; j < n; ++j) {
        double v = sin((double)(seed + 1u) * 0.001 + (double)(j + 1) * 12.9898);
        x[j] = v == 0.0 ? 1.0 : v;
    }
    double nrm = vec_norm(x, n);
    if (nrm <= 1e-12) nrm = 1.0;
    for (int j = 0; j < n; ++j) x[j] /= nrm;

    double lam = 0.0;
    for (int it = 0; it < iters; ++it) {
        memset(y, 0, (size_t)n * sizeof(double));
        if (me > 0) {
            matvec(A, me, n, x, tmp_e);
            mat_t_vec_add(A, me, n, tmp_e, y);
        }
        if (mi > 0) {
            matvec(C, mi, n, x, tmp_i);
            mat_t_vec_add(C, mi, n, tmp_i, y);
        }
        lam = 0.0;
        for (int j = 0; j < n; ++j) lam += x[j] * y[j];
        nrm = vec_norm(y, n);
        if (nrm <= 1e-12) break;
        for (int j = 0; j < n; ++j) x[j] = y[j] / nrm;
    }
    free(x); free(y); free(tmp_e); free(tmp_i);
    return sqrt(max2(lam, 0.0)) + 1e-12;
}

static void ruiz_equilibrate(
    double *q, double *c, double *A, double *b, double *C, double *d, double *l, double *u,
    int n, int me, int mi, int iterations,
    double *col_scale, double *eq_scale, double *ineq_scale
) {
    const double eps = 1e-6;
    double *col_update = (double *)calloc((size_t)n, sizeof(double));
    if (!col_update) return;
    for (int it = 0; it < iterations; ++it) {
        for (int j = 0; j < n; ++j) {
            double metric = fabs(q[j]);
            for (int r = 0; r < me; ++r) metric = max2(metric, fabs(A[(size_t)r * n + j]));
            for (int r = 0; r < mi; ++r) metric = max2(metric, fabs(C[(size_t)r * n + j]));
            col_update[j] = 1.0 / sqrt(max2(metric, eps));
        }
        for (int j = 0; j < n; ++j) {
            const double s = col_update[j];
            q[j] *= s * s;
            c[j] *= s;
            l[j] /= s;
            u[j] /= s;
            col_scale[j] *= s;
        }
        for (int r = 0; r < me; ++r)
            for (int j = 0; j < n; ++j)
                A[(size_t)r * n + j] *= col_update[j];
        for (int r = 0; r < mi; ++r)
            for (int j = 0; j < n; ++j)
                C[(size_t)r * n + j] *= col_update[j];

        for (int r = 0; r < me; ++r) {
            double metric = 0.0;
            for (int j = 0; j < n; ++j) metric = max2(metric, fabs(A[(size_t)r * n + j]));
            const double s = 1.0 / sqrt(max2(metric, eps));
            for (int j = 0; j < n; ++j) A[(size_t)r * n + j] *= s;
            b[r] *= s;
            eq_scale[r] *= s;
        }
        for (int r = 0; r < mi; ++r) {
            double metric = 0.0;
            for (int j = 0; j < n; ++j) metric = max2(metric, fabs(C[(size_t)r * n + j]));
            const double s = 1.0 / sqrt(max2(metric, eps));
            for (int j = 0; j < n; ++j) C[(size_t)r * n + j] *= s;
            d[r] *= s;
            ineq_scale[r] *= s;
        }
    }
    free(col_update);
}

static double primal_residual(const double *z, const double *A, const double *b, const double *C, const double *d, const double *l, const double *u, int n, int me, int mi) {
    double out = 0.0;
    for (int r = 0; r < me; ++r) {
        double acc = -b[r];
        for (int j = 0; j < n; ++j) acc += A[(size_t)r * n + j] * z[j];
        out = max2(out, fabs(acc));
    }
    for (int r = 0; r < mi; ++r) {
        double acc = -d[r];
        for (int j = 0; j < n; ++j) acc += C[(size_t)r * n + j] * z[j];
        out = max2(out, max2(acc, 0.0));
    }
    for (int j = 0; j < n; ++j) {
        out = max2(out, max2(l[j] - z[j], 0.0));
        out = max2(out, max2(z[j] - u[j], 0.0));
    }
    return out;
}

static double primal_dual_gap(const double *q, const double *c, const double *A, const double *b, const double *C, const double *d, const double *l, const double *u, const double *z, const double *lam, const double *mu, int n, int me, int mi) {
    double primal = 0.0;
    for (int j = 0; j < n; ++j) primal += 0.5 * q[j] * z[j] * z[j] + c[j] * z[j];
    double dual = 0.0;
    for (int r = 0; r < me; ++r) dual -= b[r] * lam[r];
    for (int r = 0; r < mi; ++r) dual -= d[r] * mu[r];
    for (int j = 0; j < n; ++j) {
        double t = c[j];
        for (int r = 0; r < me; ++r) t += A[(size_t)r * n + j] * lam[r];
        for (int r = 0; r < mi; ++r) t += C[(size_t)r * n + j] * mu[r];
        double y_dual;
        if (q[j] > 1e-12) {
            y_dual = clip(-t / q[j], l[j], u[j]);
        } else {
            y_dual = t >= 0.0 ? l[j] : u[j];
        }
        dual += 0.5 * q[j] * y_dual * y_dual + t * y_dual;
    }
    double gap = max2(primal - dual, 0.0);
    double scale = max2(1.0, max2(fabs(primal), fabs(dual)));
    return gap / scale;
}

static int solve_one(
    int n, int me, int mi,
    const double *q_in, const double *c_in, const double *A_in, const double *b_in,
    const double *C_in, const double *d_in, const double *l_in, const double *u_in,
    const double *y0, const double *lam0, const double *mu0,
    double *y_out, double *lam_out, double *mu_out,
    int max_iter, double tol, double safety, int knorm_iters, unsigned int knorm_seed,
    int is_fixed, int use_ruiz, int ruiz_iters
) {
    double *q = (double *)malloc((size_t)n * sizeof(double));
    double *c = (double *)malloc((size_t)n * sizeof(double));
    double *A = (double *)malloc((size_t)(me > 0 ? me * n : 1) * sizeof(double));
    double *b = (double *)malloc((size_t)(me > 0 ? me : 1) * sizeof(double));
    double *C = (double *)malloc((size_t)(mi > 0 ? mi * n : 1) * sizeof(double));
    double *d = (double *)malloc((size_t)(mi > 0 ? mi : 1) * sizeof(double));
    double *l = (double *)malloc((size_t)n * sizeof(double));
    double *u = (double *)malloc((size_t)n * sizeof(double));
    double *col_scale = (double *)malloc((size_t)n * sizeof(double));
    double *eq_scale = (double *)malloc((size_t)(me > 0 ? me : 1) * sizeof(double));
    double *ineq_scale = (double *)malloc((size_t)(mi > 0 ? mi : 1) * sizeof(double));
    double *z = (double *)calloc((size_t)n, sizeof(double));
    double *z_next = (double *)calloc((size_t)n, sizeof(double));
    double *zbar = (double *)calloc((size_t)n, sizeof(double));
    double *lam = (double *)calloc((size_t)(me > 0 ? me : 1), sizeof(double));
    double *lam_hat = (double *)calloc((size_t)(me > 0 ? me : 1), sizeof(double));
    double *mu = (double *)calloc((size_t)(mi > 0 ? mi : 1), sizeof(double));
    double *mu_hat = (double *)calloc((size_t)(mi > 0 ? mi : 1), sizeof(double));
    double *grad = (double *)calloc((size_t)n, sizeof(double));
    double *P = (double *)calloc((size_t)n, sizeof(double));
    if (!q || !c || !A || !b || !C || !d || !l || !u || !col_scale || !eq_scale || !ineq_scale || !z || !z_next || !zbar || !lam || !lam_hat || !mu || !mu_hat || !grad || !P) return 2;

    memcpy(q, q_in, (size_t)n * sizeof(double));
    memcpy(c, c_in, (size_t)n * sizeof(double));
    if (me > 0) {
        memcpy(A, A_in, (size_t)me * n * sizeof(double));
        memcpy(b, b_in, (size_t)me * sizeof(double));
    }
    if (mi > 0) {
        memcpy(C, C_in, (size_t)mi * n * sizeof(double));
        memcpy(d, d_in, (size_t)mi * sizeof(double));
    }
    memcpy(l, l_in, (size_t)n * sizeof(double));
    memcpy(u, u_in, (size_t)n * sizeof(double));
    for (int j = 0; j < n; ++j) col_scale[j] = 1.0;
    for (int r = 0; r < me; ++r) eq_scale[r] = 1.0;
    for (int r = 0; r < mi; ++r) ineq_scale[r] = 1.0;

    if (use_ruiz) ruiz_equilibrate(q, c, A, b, C, d, l, u, n, me, mi, ruiz_iters, col_scale, eq_scale, ineq_scale);

    double Lf = 0.0;
    double gamma = n > 0 ? q[0] : 0.0;
    for (int j = 0; j < n; ++j) {
        Lf = max2(Lf, q[j]);
        gamma = min2(gamma, q[j]);
    }
    gamma = max2(gamma, 0.0);
    double L = max2(estimate_ac_norm(A, me, C, mi, n, knorm_iters, knorm_seed), 1e-12);
    double sigma = safety / L;
    double tau = (!is_fixed && gamma > 0.0) ? safety / L : safety / (Lf + L);
    double theta = 1.0;
    for (int j = 0; j < n; ++j) P[j] = 1.0 / (1.0 + tau * q[j]);

    for (int j = 0; j < n; ++j) {
        z[j] = y0[j] / col_scale[j];
        zbar[j] = z[j];
    }
    for (int r = 0; r < me; ++r) lam[r] = lam0[r] / (eq_scale[r] == 0.0 ? 1.0 : eq_scale[r]);
    for (int r = 0; r < mi; ++r) mu[r] = mu0[r] / (ineq_scale[r] == 0.0 ? 1.0 : ineq_scale[r]);

    for (int it = 0; it < max_iter; ++it) {
        for (int r = 0; r < me; ++r) {
            double acc = -b[r];
            for (int j = 0; j < n; ++j) acc += A[(size_t)r * n + j] * zbar[j];
            lam_hat[r] = lam[r] + sigma * acc;
        }
        for (int r = 0; r < mi; ++r) {
            double acc = -d[r];
            for (int j = 0; j < n; ++j) acc += C[(size_t)r * n + j] * zbar[j];
            mu_hat[r] = max2(mu[r] + sigma * acc, 0.0);
        }
        memset(grad, 0, (size_t)n * sizeof(double));
        if (me > 0) mat_t_vec_add(A, me, n, lam_hat, grad);
        if (mi > 0) mat_t_vec_add(C, mi, n, mu_hat, grad);
        for (int j = 0; j < n; ++j) {
            z_next[j] = P[j] * (z[j] - tau * grad[j] - tau * c[j]);
            z_next[j] = clip(z_next[j], l[j], u[j]);
        }
        for (int j = 0; j < n; ++j) zbar[j] = z_next[j] + theta * (z_next[j] - z[j]);
        double pres = primal_residual(z_next, A, b, C, d, l, u, n, me, mi);
        double gap = primal_dual_gap(q, c, A, b, C, d, l, u, z_next, lam_hat, mu_hat, n, me, mi);
        memcpy(z, z_next, (size_t)n * sizeof(double));
        if (me > 0) memcpy(lam, lam_hat, (size_t)me * sizeof(double));
        if (mi > 0) memcpy(mu, mu_hat, (size_t)mi * sizeof(double));
        if (pres <= tol && gap <= tol) break;
        if (!is_fixed && gamma > 0.0) {
            double theta_new = 1.0 / sqrt(1.0 + gamma * tau);
            tau = theta_new * tau;
            sigma = sigma / theta_new;
            theta = theta_new;
            for (int j = 0; j < n; ++j) P[j] = 1.0 / (1.0 + tau * q[j]);
        }
    }

    for (int j = 0; j < n; ++j) y_out[j] = col_scale[j] * z[j];
    for (int r = 0; r < me; ++r) lam_out[r] = eq_scale[r] * lam[r];
    for (int r = 0; r < mi; ++r) mu_out[r] = ineq_scale[r] * mu[r];

    free(q); free(c); free(A); free(b); free(C); free(d); free(l); free(u);
    free(col_scale); free(eq_scale); free(ineq_scale);
    free(z); free(z_next); free(zbar); free(lam); free(lam_hat); free(mu); free(mu_hat); free(grad); free(P);
    return 0;
}

NLPOPTNET_EXPORT int nlpoptnet_cp_solve(
    int B, int n, int me, int mi,
    const double *Q, const double *c, const double *A, const double *b,
    const double *C, const double *d, const double *l, const double *u,
    const double *y0, const double *lam0, const double *mu0,
    double *y_out, double *lam_out, double *mu_out,
    int max_iter, double tol, double safety, int knorm_iters, unsigned int knorm_seed,
    int is_fixed, int use_ruiz, int ruiz_iters
) {
    for (int batch = 0; batch < B; ++batch) {
        int rc = solve_one(
            n, me, mi,
            Q + (size_t)batch * n,
            c + (size_t)batch * n,
            A + (size_t)batch * me * n,
            b + (size_t)batch * me,
            C + (size_t)batch * mi * n,
            d + (size_t)batch * mi,
            l + (size_t)batch * n,
            u + (size_t)batch * n,
            y0 + (size_t)batch * n,
            lam0 + (size_t)batch * me,
            mu0 + (size_t)batch * mi,
            y_out + (size_t)batch * n,
            lam_out + (size_t)batch * me,
            mu_out + (size_t)batch * mi,
            max_iter, tol, safety, knorm_iters, knorm_seed, is_fixed, use_ruiz, ruiz_iters
        );
        if (rc != 0) return rc;
    }
    return 0;
}
"""


class NativeProjection:
    """ctypes wrapper around the compiled native CP projection library."""

    def __init__(self, shared_library: str | Path):
        """Load a shared library that exposes ``nlpoptnet_cp_solve``."""
        self.path = Path(shared_library).expanduser().resolve()
        self.lib = ctypes.CDLL(str(self.path))
        ptr = ctypes.POINTER(ctypes.c_double)
        self._solve = self.lib.nlpoptnet_cp_solve
        self._solve.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._solve.restype = ctypes.c_int

    @staticmethod
    def _array(value, shape: tuple[int, ...]) -> np.ndarray:
        """Convert input to a contiguous float64 array with a fixed shape."""
        arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        if arr.shape != shape:
            raise ValueError(f"Expected array shape {shape}, got {arr.shape}.")
        return arr

    def solve(
        self,
        Q_diag,
        c,
        A,
        b,
        C,
        d,
        l,
        u,
        y0,
        lam0,
        mu0,
        *,
        cfg,
    ):
        """Solve a batched projection problem with the native backend."""
        Q_diag = np.ascontiguousarray(np.asarray(Q_diag, dtype=np.float64))
        if Q_diag.ndim != 2:
            raise ValueError("Q_diag must have shape (batch, n).")
        B, n = Q_diag.shape
        c = self._array(c, (B, n))
        A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
        C = np.ascontiguousarray(np.asarray(C, dtype=np.float64))
        if A.ndim != 3 or A.shape[0] != B or A.shape[2] != n:
            raise ValueError("A must have shape (batch, me, n).")
        if C.ndim != 3 or C.shape[0] != B or C.shape[2] != n:
            raise ValueError("C must have shape (batch, mi, n).")
        me = int(A.shape[1])
        mi = int(C.shape[1])
        b = self._array(b, (B, me))
        d = self._array(d, (B, mi))
        l = self._array(l, (B, n))
        u = self._array(u, (B, n))
        y0 = self._array(y0, (B, n))
        lam0 = self._array(lam0, (B, me))
        mu0 = self._array(mu0, (B, mi))
        y_out = np.empty((B, n), dtype=np.float64)
        lam_out = np.empty((B, me), dtype=np.float64)
        mu_out = np.empty((B, mi), dtype=np.float64)
        ptr = ctypes.POINTER(ctypes.c_double)
        rc = self._solve(
            B,
            n,
            me,
            mi,
            Q_diag.ctypes.data_as(ptr),
            c.ctypes.data_as(ptr),
            A.ctypes.data_as(ptr),
            b.ctypes.data_as(ptr),
            C.ctypes.data_as(ptr),
            d.ctypes.data_as(ptr),
            l.ctypes.data_as(ptr),
            u.ctypes.data_as(ptr),
            y0.ctypes.data_as(ptr),
            lam0.ctypes.data_as(ptr),
            mu0.ctypes.data_as(ptr),
            y_out.ctypes.data_as(ptr),
            lam_out.ctypes.data_as(ptr),
            mu_out.ctypes.data_as(ptr),
            int(cfg.cp_iters),
            float(cfg.cp_tol),
            float(cfg.safety),
            int(cfg.knorm_iters),
            int(cfg.knorm_seed),
            1 if bool(cfg.IS_FIXED) else 0,
            1 if bool(cfg.use_ruiz) else 0,
            int(cfg.ruiz_iters),
        )
        if rc != 0:
            raise RuntimeError(f"Native projection solver failed with code {rc}.")
        return y_out, lam_out, mu_out


def _platform_tag() -> str:
    """Return a platform tag used for caching compiled libraries."""
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower() or "unknown"
    return f"{system}-{machine}"


def _shared_library_name() -> str:
    """Return the platform-specific shared library filename."""
    system = platform.system().lower()
    if system == "windows":
        return "projection_native.dll"
    if system == "darwin":
        return "projection_native.dylib"
    return "projection_native.so"


def _native_cache_root() -> Path:
    """Return the root directory used to cache native projection binaries."""
    override = os.environ.get("NLPOPTNET_NATIVE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    system = platform.system().lower()
    if system == "windows":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(Path.home())
        return Path(base) / "nlpoptnet" / "native_projection"
    if system == "darwin":
        return Path.home() / "Library" / "Caches" / "nlpoptnet" / "native_projection"
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "nlpoptnet" / "native_projection"
    return Path.home() / ".cache" / "nlpoptnet" / "native_projection"


def _native_cache_dir() -> Path:
    """Return the legacy versioned cache directory for older run manifests."""
    return _native_cache_root() / _SOURCE_VERSION / _platform_tag()


def _native_artifact_dir(run_dir: str | Path, *, platform_tag: str | None = None) -> Path:
    """Return the run-local directory used to store native projection binaries."""
    tag = str(platform_tag or _platform_tag())
    return Path(run_dir) / "native_projection" / _SOURCE_VERSION / tag


def _native_library_path(run_dir: str | Path, *, platform_tag: str | None = None) -> Path:
    """Return the run-local shared library path for the current platform."""
    return _native_artifact_dir(run_dir, platform_tag=platform_tag) / _shared_library_name()


def _compiler_candidates(explicit: str | None = None) -> list[str]:
    """Return candidate C compilers to try for native compilation."""
    if explicit is not None:
        return [explicit]
    candidates: list[str] = []
    env_cc = os.environ.get("CC")
    if env_cc:
        candidates.append(env_cc)
    if platform.system().lower() == "windows":
        candidates.extend(["cl", "gcc", "clang"])
    else:
        candidates.extend(["cc", "gcc", "clang"])
    out: list[str] = []
    for candidate in candidates:
        if candidate in out:
            continue
        if Path(candidate).is_absolute() or shutil.which(candidate) is not None:
            out.append(candidate)
    return out


def _compile_command(compiler: str, source: Path, shared: Path) -> list[str]:
    """Build the platform-specific compile command."""
    system = platform.system().lower()
    compiler_name = Path(compiler).name.lower()
    if system == "windows" and compiler_name in {"cl", "cl.exe"}:
        return [compiler, "/O2", "/LD", str(source), f"/Fe:{shared}"]
    if system == "darwin":
        return [compiler, "-O3", "-std=c99", "-fPIC", "-dynamiclib", str(source), "-lm", "-o", str(shared)]
    return [compiler, "-O3", "-std=c99", "-fPIC", "-shared", str(source), "-lm", "-o", str(shared)]


def _platform_payload(
    *,
    source: str | None = None,
    shared_library: str | None = None,
    status: str,
    compiler: str | None = None,
    command: list[str] | None = None,
    returncode: int | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> dict[str, Any]:
    """Create a normalized payload for one platform-specific native artifact."""
    return {
        "source_version": _SOURCE_VERSION,
        "source": source,
        "source_stored": False,
        "shared_library": shared_library,
        "platform": platform.system(),
        "machine": platform.machine(),
        "platform_tag": _platform_tag(),
        "compiler": compiler,
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "status": status,
    }


def _manifest_payload(
    *,
    current: dict[str, Any],
    artifacts: dict[str, dict[str, Any]] | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Create a normalized manifest payload for native compilation state."""
    platform_tag = str(current.get("platform_tag", _platform_tag()))
    stored_artifacts = {
        str(tag): dict(entry)
        for tag, entry in (artifacts or {}).items()
        if isinstance(entry, dict)
    }
    stored_artifacts[platform_tag] = dict(current)
    payload = {
        "format": _NATIVE_FORMAT,
        "source_version": _SOURCE_VERSION,
        "source": current.get("source"),
        "source_stored": bool(current.get("source_stored", False)),
        "shared_library": current.get("shared_library"),
        "cache_scope": "run",
        "platform": current.get("platform"),
        "machine": current.get("machine"),
        "platform_tag": platform_tag,
        "batch_mode": "sequential",
        "compiler": current.get("compiler"),
        "command": current.get("command"),
        "returncode": current.get("returncode"),
        "stdout": current.get("stdout"),
        "stderr": current.get("stderr"),
        "status": current.get("status"),
        "artifacts": stored_artifacts,
    }
    if message is not None:
        payload["message"] = message
    return payload


def _read_manifest(path: Path) -> dict[str, Any] | None:
    """Load a compilation manifest if it exists and is valid JSON."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _artifact_entries(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return platform-specific artifact entries from current or legacy manifests."""
    if payload is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for tag, entry in artifacts.items():
            if not isinstance(entry, dict):
                continue
            normalized = dict(entry)
            normalized.setdefault("platform_tag", str(tag))
            normalized.setdefault("source_version", payload.get("source_version", _SOURCE_VERSION))
            out[str(normalized["platform_tag"])] = normalized
    legacy_tag = payload.get("platform_tag")
    if legacy_tag is not None and str(legacy_tag) not in out:
        legacy_entry = {
            key: payload.get(key)
            for key in (
                "source_version",
                "source",
                "source_stored",
                "shared_library",
                "platform",
                "machine",
                "platform_tag",
                "compiler",
                "command",
                "returncode",
                "stdout",
                "stderr",
                "status",
            )
            if key in payload
        }
        legacy_entry.setdefault("source_version", payload.get("source_version", _SOURCE_VERSION))
        legacy_entry.setdefault("platform_tag", str(legacy_tag))
        out[str(legacy_tag)] = legacy_entry
    return out


def _resolve_entry_shared_library(entry: dict[str, Any] | None, run_dir: Path) -> Path | None:
    """Resolve a shared library path for one platform entry."""
    if entry is None:
        return None
    if entry.get("status") != "ok":
        return None
    if entry.get("source_version") != _SOURCE_VERSION:
        return None
    shared_library = entry.get("shared_library")
    if not shared_library:
        return None

    candidate = Path(str(shared_library)).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    run_local = run_dir / candidate
    if run_local.exists():
        return run_local

    legacy_relative = run_dir / str(shared_library)
    if legacy_relative.exists():
        return legacy_relative

    if entry.get("platform_tag") == _platform_tag():
        legacy_cached = _native_cache_dir() / candidate.name
        if legacy_cached.exists():
            return legacy_cached
    return None


def _is_usable_manifest(payload: dict[str, Any] | None, run_dir: Path) -> bool:
    """Check whether a manifest points to a compatible shared library."""
    return _resolve_shared_library(payload, run_dir) is not None


def _resolve_shared_library(payload: dict[str, Any] | None, run_dir: Path) -> Path | None:
    """Resolve the actual shared library path from a manifest payload."""
    current = _artifact_entries(payload).get(_platform_tag())
    return _resolve_entry_shared_library(current, run_dir)


def _recompile_message(payload: dict[str, Any] | None, run_dir: Path) -> str | None:
    """Describe why the current system needs a fresh native compilation."""
    if payload is None:
        return None
    artifacts = _artifact_entries(payload)
    current = artifacts.get(_platform_tag())
    if current is not None:
        if current.get("source_version") != _SOURCE_VERSION:
            return "Native projection artifact is out of date. Compiling for current system."
        if _resolve_entry_shared_library(current, run_dir) is None:
            return "Native projection artifact for the current system was not found. Compiling for current system."
        return None
    if artifacts:
        return "Model was trained on a different system. Compiling for current system."
    return "Native projection artifact was not found. Compiling for current system."


def _relative_run_path(path: Path, run_dir: Path) -> str:
    """Return a manifest-friendly relative path when possible."""
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return str(path)


def compile_native_projection(run_dir: str | Path, *, cc: str | None = None, force: bool = False) -> dict[str, Any]:
    """Compile and store the native projection shared library in the run directory."""
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "projection_native.json"
    existing = _read_manifest(manifest)
    if not force and _is_usable_manifest(existing, target):
        return existing

    artifact_dir = _native_artifact_dir(target)
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        current = _platform_payload(status="cache-unavailable", stderr=f"{type(exc).__name__}: {exc}")
        payload = _manifest_payload(
            current=current,
            artifacts=_artifact_entries(existing),
            message=_recompile_message(existing, target),
        )
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload
    shared = _native_library_path(target)
    compilers = _compiler_candidates(cc)
    if not compilers:
        current = _platform_payload(status="missing-compiler")
        payload = _manifest_payload(
            current=current,
            artifacts=_artifact_entries(existing),
            message=_recompile_message(existing, target),
        )
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    message = _recompile_message(existing, target)
    artifacts = _artifact_entries(existing)
    last_payload: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="nlpoptnet_native_") as tmp:
        source = Path(tmp) / "projection_native.c"
        source.write_text(_C_SOURCE, encoding="utf-8")
        for compiler in compilers:
            if shared.exists():
                try:
                    shared.unlink()
                except OSError:
                    pass
            cmd = _compile_command(compiler, source, shared)
            try:
                proc = subprocess.run(cmd, cwd=str(target), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            except OSError as exc:
                payload = _platform_payload(
                    source=None,
                    status="compile-failed",
                    compiler=compiler,
                    command=cmd,
                    returncode=None,
                    stdout=None,
                    stderr=f"{type(exc).__name__}: {exc}",
                )
                last_payload = payload
                continue
            payload = _platform_payload(
                source=None,
                shared_library=_relative_run_path(shared, target) if proc.returncode == 0 and shared.exists() else None,
                status="ok" if proc.returncode == 0 and shared.exists() else "compile-failed",
                compiler=compiler,
                command=cmd,
                returncode=int(proc.returncode),
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
            last_payload = payload
            if payload["status"] == "ok":
                break

    current = last_payload if last_payload is not None else _platform_payload(status="compile-failed")
    payload = _manifest_payload(current=current, artifacts=artifacts, message=message)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_or_compile_native_projection(run_dir: str | Path) -> tuple[NativeProjection | None, dict[str, Any]]:
    """Load a usable native projection library or try to compile one."""
    target = Path(run_dir)
    manifest_path = target / "projection_native.json"
    manifest = _read_manifest(manifest_path)
    if not _is_usable_manifest(manifest, target):
        manifest = compile_native_projection(target)
    if _is_usable_manifest(manifest, target):
        try:
            shared = _resolve_shared_library(manifest, target)
            if shared is None:
                raise FileNotFoundError("Native projection library is not available.")
            return NativeProjection(shared), manifest
        except Exception as exc:
            current = _platform_payload(status="load-failed", stderr=f"{type(exc).__name__}: {exc}")
            failed = _manifest_payload(
                current=current,
                artifacts=_artifact_entries(manifest),
                message=manifest.get("message") if isinstance(manifest, dict) else None,
            )
            manifest_path.write_text(json.dumps(failed, indent=2, sort_keys=True), encoding="utf-8")
            return None, failed
    if manifest is not None:
        return None, manifest
    missing = _manifest_payload(current=_platform_payload(status="missing"))
    return None, missing
