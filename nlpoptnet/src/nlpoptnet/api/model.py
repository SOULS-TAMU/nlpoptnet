"""Public NLPOptNet interface over shared learning and projection engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    import numpy as np

from typing import Any
import jax
import copy
import nlpoptnet.api.configuration as _m0
import nlpoptnet.api.data as _m1
import nlpoptnet.api.deployment as _m2
import nlpoptnet.api.fit as _m3
import nlpoptnet.api.formulation.builder as _m4
import nlpoptnet.api.formulation.expressions as _m5
import nlpoptnet.api.formulation.scaling as _m6
import nlpoptnet.api.predict as _m7
import nlpoptnet.api.problem as _m8
import nlpoptnet.api.reporting as _m9
import nlpoptnet.serialization.definition as _m10
from nlpoptnet.api.configuration import _normalize_problem_type
from nlpoptnet.api.configuration import _DEFAULT_CONFIG
from nlpoptnet.api.configuration import _apply_config_aliases
from nlpoptnet.api.configuration import _BuildState
from nlpoptnet.api.configuration import _InferenceState
from nlpoptnet.api.configuration import _ScalingState
from nlpoptnet.api.formulation.expressions import _SymbolNamespace
from nlpoptnet.api.formulation.expressions import _ConstraintManager
from nlpoptnet.api.formulation.expressions import Expression

jax.config.update("jax_enable_x64", True)


class NLPOptNet:
    """Define a nonlinear problem and supply its resources to OHAGI and PURIN."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        type: str | None = None,
        name: str | None = None,
    ) -> None:
        self.problem_type = _normalize_problem_type(type)
        self.name = str(name or (self.problem_type or "nlpoptnet"))
        self.config = copy.deepcopy(_DEFAULT_CONFIG)
        if config is not None:
            self.config.update(copy.deepcopy(dict(config)))
        _apply_config_aliases(self.config)

        self.parameter_names: list[str] = []
        self.variable_names: list[str] = []
        self.parameter = _SymbolNamespace(self, "parameter")
        self.variable = _SymbolNamespace(self, "variable")
        self.parameters = self.parameter
        self.variables = self.variable
        self.constraints = _ConstraintManager(self)

        self._objective_expr: Expression | None = None
        self._objective_callable: Callable | None = None
        self._constants: dict[str, np.ndarray] = {}
        self._constant_counter = 0
        self._dataset_spec: dict[str, Any] | None = None
        self._region_spec: dict[str, Any] | None = None
        self._build_state: _BuildState | None = None
        self._inference_state: _InferenceState | None = None
        self._metadata_path: str | None = None
        self._extracted_problem_path: str | None = None
        self.scaling = _ScalingState()

    dataset = _m1.dataset
    simplex = _m1.simplex
    box = _m1.box
    parameter_region = _m1.parameter_region
    _load_or_sample_parameters = _m1._load_or_sample_parameters
    _normalize_region_constraint = _m1._normalize_region_constraint
    _resolve_polytope_matrix = _m1._resolve_polytope_matrix
    _resolve_num_samples = _m1._resolve_num_samples
    _train_cfg_dict = _m0._train_cfg_dict
    build = _m8.build
    optimize = _m3.optimize
    predict = _m7.predict
    _project_with_state = _m7._project_with_state
    export_inference = _m2.export_inference
    load_inference = staticmethod(_m2.load_inference)
    summary = _m9.summary
    plot_history = _m9.plot_history
    _run_dir = _m9._run_dir
    _read_history = _m9._read_history
    _history_column = staticmethod(_m9._history_column)
    _estimate_inference_times = _m9._estimate_inference_times
    _write_history_csv = _m9._write_history_csv
    load = _m10.load
    _metadata_config = _m10._metadata_config
    _problem_is_serializable = _m10._problem_is_serializable
    scale = _m6.scale
    _positive_scale = _m6._positive_scale
    _positive_scalar_scale = _m6._positive_scalar_scale
    _scaled_parameters = _m6._scaled_parameters
    _unscale_variables = _m6._unscale_variables
    _eval_affine_bounds_numpy = _m6._eval_affine_bounds_numpy
    _estimate_variable_scale_from_linear_inequalities = (
        _m6._estimate_variable_scale_from_linear_inequalities
    )
    _estimate_variable_scale_from_sensitivities = (
        _m6._estimate_variable_scale_from_sensitivities
    )
    _compute_basic_scales = _m6._compute_basic_scales
    _estimate_linear_objective_scale = _m6._estimate_linear_objective_scale
    _estimate_objective_and_constraint_scales = (
        _m6._estimate_objective_and_constraint_scales
    )
    _elementwise = _m5._elementwise
    _register_constant = _m5._register_constant
    _ensure_constant = _m5._ensure_constant
    _constant_to_scalar_expr = _m5._constant_to_scalar_expr
    _constant_to_vector_expr = _m5._constant_to_vector_expr
    _zero_scalar = _m5._zero_scalar
    _zero_vector = _m5._zero_vector
    _as_scalar_expr = _m5._as_scalar_expr
    _as_vector_expr = _m5._as_vector_expr
    _as_affine_vector = _m5._as_affine_vector
    _combine_scalar_affine = _m5._combine_scalar_affine
    _combine_vector_affine = _m5._combine_vector_affine
    _check_symbol = _m5._check_symbol
    add_parameter = _m4.add_parameter
    add_variable = _m4.add_variable
    objective = _m4.objective
    matrix = _m4.matrix
    vector = _m4.vector
    tensor = _m4.tensor
    extract = _m4.extract
    lin = _m4.lin
    batch_lin = _m4.batch_lin
    quad = _m4.quad
    batch_quad = _m4.batch_quad
    batch_exp = _m4.batch_exp
    sin = _m4.sin
    cos = _m4.cos
    exp = _m4.exp
    log = _m4.log
    sqrt = _m4.sqrt
    abs = _m4.abs
    _build_box_bounds = _m4._build_box_bounds


ProblemBuilder = NLPOptNet

# Resolve public forward annotations after the model class is constructed.
for _adapter in (
    _m0,
    _m1,
    _m2,
    _m3,
    _m4,
    _m5,
    _m6,
    _m7,
    _m8,
    _m9,
    _m10,
):
    _adapter.NLPOptNet = NLPOptNet
del _adapter
