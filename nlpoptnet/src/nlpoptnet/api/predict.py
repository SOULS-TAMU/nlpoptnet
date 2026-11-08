"""Predict resources for the NLPOptNet frontend."""

from __future__ import annotations
from jax import block_until_ready as _block_until_ready
import jax
import jax.numpy as jnp
import numpy as np
from nlpoptnet.api.projection import projected_model as projected_model
from ohagi.execution.inference import require_jax_backend
from purin.execution.runtime import resolve_jax_dtype as resolve_dtype


def predict(self, values, *, projection_backend: str = "auto") -> np.ndarray:
    """Predict projected variables for one sample or a batch of samples."""
    if self._inference_state is None:
        raise RuntimeError("Please train or load the model before calling predict().")
    data = np.asarray(values, dtype=np.float64)
    squeeze = data.ndim == 1
    if squeeze:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] != self._inference_state.n_x:
        raise ValueError(
            f"predict expected a 1D or 2D array with {self._inference_state.n_x} parameter values."
        )
    require_jax_backend(projection_backend)
    if self.scaling.enabled:
        assert self.scaling.D_p is not None
        data_internal = data / self.scaling.D_p.reshape(1, -1)
    else:
        data_internal = data

    x_batch = jnp.asarray(
        data_internal,
        dtype=resolve_dtype(str(self._inference_state.train_config["dtype"])),
    )
    projected = self._project_with_state(
        params=self._inference_state.params,
        backbone=self._inference_state.backbone,
        sub_layer=self._inference_state.sub_layer,
        cfg=self._inference_state.cfg,
        X=x_batch,
    )
    out = np.asarray(projected, dtype=np.float64)
    out = self._unscale_variables(out)
    return out[0] if squeeze else out


def _project_with_state(
    self,
    *,
    params,
    backbone,
    sub_layer,
    cfg,
    X,
):
    # Cache the whole inference graph, with weights as dynamic arguments.
    # This helper also evaluates updated training parameters.
    key = (id(backbone), id(sub_layer), id(cfg))
    cached = getattr(self, "_jax_prediction_cache", None)
    if cached is None or cached[0] != key:
        from nlpoptnet.api.projection import projected_model

        model = projected_model(backbone, sub_layer, cfg)

        def forward(parameters, inputs):
            outputs, _ = model.apply(parameters, {}, inputs)
            return outputs["projected"]

        cached = (key, jax.jit(forward))
        self._jax_prediction_cache = cached
    y_proj = cached[1](params, X)
    _block_until_ready(y_proj)
    return y_proj
