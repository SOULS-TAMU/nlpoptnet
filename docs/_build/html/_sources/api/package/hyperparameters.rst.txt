Hyperparameters
===============

``NLPOptNet`` accepts a configuration dictionary that is normalized into the training configuration described by :class:`opt.training.config.TrainConfig`.

Required keys
-------------

- ``epochs``
- ``batch_size``
- ``learning_rate``
- ``hidden_size``
- ``hidden_layers``

Common keys
-----------

.. list-table::
   :header-rows: 1

   * - Key
     - Meaning
   * - ``train_frac``
     - Fraction of parameter samples used for training.
   * - ``seed``
     - Random seed for splitting and initialization.
   * - ``dtype``
     - Numeric precision, typically ``float64``.
   * - ``print_every``
     - Console logging frequency during training.
   * - ``verbose``
     - Whether to print progress messages.

Projection and optimization keys
--------------------------------

.. list-table::
   :header-rows: 1

   * - Key
     - Meaning
   * - ``alpha_consistency``
     - Weight on the projection consistency penalty.
   * - ``cp_mode``
     - Projection mode, ``fixed`` or ``accelerated``.
   * - ``cp_iters``
     - Maximum Chambolle-Pock iterations.
   * - ``cp_tol``
     - Solver tolerance.
   * - ``safety``
     - Step-size safety factor.
   * - ``knorm_iters``
     - Power-iteration count used in operator norm estimation.
   * - ``knorm_seed``
     - Seed used for the norm-estimation path.
   * - ``adjoint_iters``
     - Iteration budget for implicit backward solves.
   * - ``k_layer``
     - Number of projection layers applied after the backbone.
   * - ``use_ruiz``
     - Whether to apply Ruiz equilibration.
   * - ``ruiz_iters``
     - Number of Ruiz scaling iterations.
   * - ``jit_warmup``
     - Whether to run an initial compile warmup pass.
   * - ``native_projection``
     - Prefer native projection artifacts when they are available.

Sampling and bounds
-------------------

.. list-table::
   :header-rows: 1

   * - Key
     - Meaning
   * - ``num_samples``
     - Default number of parameter samples to generate for sampled regions.
   * - ``y_bound``
     - Default fallback magnitude for variable bounds when explicit box bounds are absent.

Core code reference
-------------------

.. autoclass:: opt.training.config.TrainConfig
   :members:
   :exclude-members: batch_size, epochs, learning_rate, alpha_consistency, train_frac, val_frac, hidden_size, hidden_dim, cp_iters, cp_tol, cp_mode, IS_FIXED, stepsize, safety, knorm_iters, knorm_seed, seed, adjoint_iters, use_ruiz, ruiz_iters, k_layer, dtype, device, jit_warmup
   :member-order: bysource

.. autofunction:: opt.training.config.cfg_from_dict
