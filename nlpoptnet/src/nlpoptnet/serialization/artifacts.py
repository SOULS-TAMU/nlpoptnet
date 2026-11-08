"""Artifacts resources for the NLPOptNet frontend."""

from __future__ import annotations
from nlpoptnet.api.formulation.expressions import Constraint
from pathlib import Path
from nlpoptnet.api.formulation.expressions import _BlockConstraint
from nlpoptnet.api.configuration import _InferenceState
import jax
import jax.numpy as jnp
import numpy as np
from nlpoptnet.api.projection import projected_model as projected_model
from purin.execution.runtime import resolve_device
from purin.execution.runtime import resolve_jax_dtype
from ohagi._utils.serialization import resolve_path
import shutil
from ohagi._utils.serialization import timestamp
from ohagi.data.sampling import write_csv_matrix
from ohagi._utils.serialization import write_json


def save_run(self, *, build_state, state, history, summary, predictions_scaled):
    """Write NLPOPTNet run reports and the shared OHAGI model bundle."""
    run_dir = Path.cwd() / f"{self.name}_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    parameters_path = run_dir / "parameters.csv"
    predictions_path = run_dir / "predicted_variables.csv"
    history_path = run_dir / "history.csv"
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    constants_path = run_dir / "problem_constants.npz"

    predictions_original = self._unscale_variables(
        np.asarray(predictions_scaled, dtype=np.float64)
    )
    write_csv_matrix(
        parameters_path, build_state.X_original, headers=self.parameter_names
    )
    write_csv_matrix(
        predictions_path, predictions_original, headers=self.variable_names
    )
    self._write_history_csv(history_path, history)
    np.savez(constants_path, **self._constants)
    if self._extracted_problem_path is not None:
        shutil.copy2(
            resolve_path(self._extracted_problem_path), run_dir / "problem.npz"
        )

    lower_M = build_state.lower_M
    lower_c = build_state.lower_c
    upper_M = build_state.upper_M
    upper_c = build_state.upper_c

    metadata = {
        "version": "0.2.0",
        "model_name": self.name,
        "created_at": timestamp(),
        "problem_type": self.problem_type,
        "config": self._metadata_config(build_state.train_config),
        "problem": {
            "serializable": self._problem_is_serializable(),
            "parameter_names": list(self.parameter_names),
            "variable_names": list(self.variable_names),
            "objective_text": None
            if self._objective_expr is None
            else self._objective_expr.text,
            "equality_texts": [
                entry.text
                for entry in self.constraints.equality.items
                if isinstance(entry, Constraint)
            ],
            "inequality_texts": [
                entry.text
                for entry in self.constraints.inequality.items
                if isinstance(entry, Constraint)
            ],
            "bounds": {
                "lower_M": lower_M.tolist(),
                "lower_c": lower_c.tolist(),
                "upper_M": upper_M.tolist(),
                "upper_c": upper_c.tolist(),
            },
            "scaling": {
                "enabled": bool(self.scaling.enabled),
                "D_p": None
                if self.scaling.D_p is None
                else np.asarray(self.scaling.D_p, dtype=np.float64).tolist(),
                "D_v": None
                if self.scaling.D_v is None
                else np.asarray(self.scaling.D_v, dtype=np.float64).tolist(),
                "D_obj": float(self.scaling.D_obj),
                "D_eq": None
                if self.scaling.D_eq is None
                else np.asarray(self.scaling.D_eq, dtype=np.float64)
                .reshape(-1)
                .tolist(),
                "D_ineq": None
                if self.scaling.D_ineq is None
                else np.asarray(self.scaling.D_ineq, dtype=np.float64)
                .reshape(-1)
                .tolist(),
                "mode": str(self.scaling.mode),
                "use_jacobian_scaling": bool(self.scaling.use_jacobian_scaling),
                "probe_points": int(self.scaling.probe_points),
                "objective_percentile": float(self.scaling.objective_percentile),
                "constraint_percentile": float(self.scaling.constraint_percentile),
                "eps": float(self.scaling.eps),
            },
            "has_callable_objective": self._objective_callable is not None,
            "has_callable_blocks": any(
                isinstance(entry, _BlockConstraint)
                for entry in self.constraints.equality.items
                + self.constraints.inequality.items
            ),
        },
        "artifacts": {
            "parameters": parameters_path.name,
            "predicted_variables": predictions_path.name,
            "history": history_path.name,
            "summary": summary_path.name,
            "constants": constants_path.name,
            "problem_npz": "problem.npz"
            if self._extracted_problem_path is not None
            else None,
        },
    }
    self._metadata_path = str(metadata_path)
    self._inference_state = _InferenceState(
        model=build_state.model,
        cfg=build_state.cfg,
        train_config=build_state.train_config,
        params=state.params,
        backbone=build_state.backbone,
        sub_layer=build_state.sub_layer,
        n_x=len(self.parameter_names),
        output_dir=str(run_dir),
    )
    from ohagi.serialization import save_model
    from nlpoptnet.serialization.artifacts import nlpoptnet_graph

    save_model(
        metadata_path,
        framework="nlpoptnet",
        parameters=state.params,
        definition=metadata,
        inference_spec=nlpoptnet_graph(self),
    )

    summary.update(
        self._estimate_inference_times(
            metadata_path=metadata_path,
            X_train=np.asarray(build_state.X[build_state.train_idx], dtype=np.float64),
            batch_size=int(build_state.cfg.batch_size),
        )
    )

    if bool(self.config.get("verbose", True)):
        print("Done.")

    write_json(summary_path, summary)

    msg = (
        "\n" + "=" * 120 + "\n"
        "If you use this model in your research, please cite:\n"
        "Nath Roy, Golder, & Hasan (2026) NLPOpt-Net: A Learning Method for Nonlinear Optimization with Feasibility Guarantees.\n"
        "https://github.com/SOULS-TAMU/nlpoptnet\n"
        "Contact: bimolnathroy@tamu.edu, rahulgolder8420@tamu.edu, hasan@tamu.edu\n"
        + "=" * 120
        + "\n"
    )
    print(msg)

    return {
        "output_dir": str(run_dir),
        "metadata_path": str(metadata_path),
        "summary": summary,
        "history": history,
    }


def nlpoptnet_graph(model, *, device="cpu"):
    """Capture trained weights, scaling and nonlinear correction in a pure JAX graph."""
    from nlpoptnet.api.projection import projected_model

    state = model._inference_state
    if state is None:
        raise RuntimeError("Train or load NLPOPTNet before exporting inference.")
    selected = resolve_device(device)
    params = jax.device_put(state.params, selected)
    dtype = resolve_jax_dtype(state.train_config["dtype"])
    scaling = bool(model.scaling.enabled)
    dp = jnp.asarray(model.scaling.D_p, dtype=dtype) if scaling else 1.0
    dv = jnp.asarray(model.scaling.D_v, dtype=dtype) if scaling else 1.0
    backbone, layer, cfg = state.backbone, state.sub_layer, state.cfg
    shared_model = projected_model(backbone, layer, cfg)

    def graph(inputs):
        internal = inputs / dp
        outputs, _ = shared_model.apply(params, {}, internal)
        return outputs["projected"] * dv

    return dict(
        predict_fn=graph,
        framework="nlpoptnet",
        input_size=state.n_x,
        output_size=len(model.variable_names),
        dtype=np.dtype(dtype).name,
        input_names=list(model.parameter_names),
        output_names=list(model.variable_names),
    )
