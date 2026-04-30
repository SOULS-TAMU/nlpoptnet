Configuration Options
=====================

The following configuration options are available in NLPOpt-Net:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - Default Value
     - Required
   * - ``epochs``
     - Number of training epochs
     - —
     - Yes
   * - ``batch_size``
     - Number of samples per training batch
     - —
     - Yes
   * - ``learning_rate``
     - Learning rate for optimizer
     - —
     - Yes
   * - ``hidden_size``
     - Width of hidden layers in the neural network
     - —
     - Yes
   * - ``hidden_layers``
     - Number of hidden layers in the neural network
     - —
     - Yes
   * - ``train_frac``
     - Fraction of data used for training (rest for validation)
     - 0.8
     - No
   * - ``seed``
     - Random seed for reproducibility
     - 42
     - No
   * - ``alpha_consistency``
     - Weight for consistency loss in training
     - 10.0
     - No
   * - ``cp_mode``
     - Chambolle–Pock mode: ``'fixed'`` or ``'accelerated'``
     - ``'fixed'``
     - No
   * - ``cp_iters``
     - Number of iterations for CP projection solver
     - 500
     - No
   * - ``cp_tol``
     - Tolerance for convergence of CP solver
     - 1e-9
     - No
   * - ``safety``
     - Safety factor for projection stability
     - 0.95
     - No
   * - ``knorm_iters``
     - Iterations for estimating operator norm (K-norm)
     - 25
     - No
   * - ``knorm_seed``
     - Seed for K-norm estimation
     - 42
     - No
   * - ``adjoint_iters``
     - Iterations for implicit differentiation (adjoint solve)
     - 25
     - No
   * - ``k_layer``
     - Number of projection layers applied
     - 1
     - No
   * - ``use_ruiz``
     - Whether to apply Ruiz equilibration
     - True
     - No
   * - ``ruiz_iters``
     - Number of Ruiz scaling iterations
     - 10
     - No
   * - ``dtype``
     - Numerical precision (``float32`` or ``float64``)
     - ``'float64'``
     - No
   * - ``print_every``
     - Frequency of printing training logs
     - 50
     - No
   * - ``device``
     - Device to run on (``cpu``, ``gpu``, or ``auto``)
     - ``'auto'``
     - No
   * - ``verbose``
     - Whether to print detailed logs
     - True
     - No

NLPOpt-Net takes a dictionary input for configurations. Following is an example of defining the configuration options and creating a model object.

.. code-block:: python

   from nlpoptnet import NLPOptNet

   CONFIG = {
       'epochs': 1000,
       'batch_size': 40,
       'learning_rate': 1e-3,
       'train_frac': 0.5,
       'hidden_size': 64,
       'hidden_layers': 2,
       'seed': 42,
       'alpha_consistency': 10.0,
       'cp_mode': 'fixed',
       'cp_iters': 300,
       'cp_tol': 1e-09,
       'safety': 0.95,
       'knorm_iters': 15,
       'knorm_seed': 42,
       'adjoint_iters': 20,
       'k_layer': 1,
       'use_ruiz': True,
       'ruiz_iters': 5,
       'dtype': 'float64',
       'print_every': 100,
       'device': 'auto',
       'verbose': True,
   }

   model = NLPOptNet(config=CONFIG, type='qp', name='Example_QP_Model')

Here, the ``CONFIG`` dictionary defines the configuration of the model.
Users must provide all required configuration options. For optional
parameters, if not specified, default values will be used.

The ``type`` argument is used for reference and bookkeeping purposes
only and does not affect any computational procedure. Users may set
it to ``None``; however, it is recommended to specify the problem type
for better experiment tracking and reproducibility. The allowed values
are ``qp``, ``qcqp``, ``nlp``, ``nonconvex``, or ``None``.

The ``name`` argument is used to assign a custom name to the model,
which is also used for naming the output directory where results
are saved. The naming behavior is as follows:

- If ``name`` is provided → output directory uses ``name``
- Else if ``type`` is provided → output directory uses ``type``
- Else → output directory defaults to ``nlpoptnet``

The output directory is named using the pattern:

``<model_name>_<date>_<time>``

where:

- ``model_name`` is determined by the ``name`` or ``type`` argument
- ``date`` and ``time`` are automatically generated timestamps

For example:

- ``Example_QP_Model_20260423_104512``
- ``qp_20260423_104512``
- ``nlpoptnet_20260423_104512``
