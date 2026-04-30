Defining Parameter Space
========================

The user must specify the parameter space from which training samples are drawn. NLPOpt-Net supports multiple ways to define this space.

- **CSV-based dataset**

  - Users can explicitly provide parameter samples via a ``.csv`` file.
  - Each row corresponds to one sample of the parameter vector.
  - This is the most flexible option when sampling data is already available.

- **Box-constrained space**

  - Users can define a box (hyper-rectangle) for the parameters.
  - ``x_lower`` and ``x_upper`` can be scalars or vectors.
  - Sampling is performed uniformly within the box.

- **Simplex / Polytope space**

  - Users can define a simplex or general polytope region.
  - The space can be defined:

    - using structured constraints, or
    - using matrix form: :math:`M x \leq 1`.

  - Useful for constrained parameter spaces (e.g., probabilities, resource allocation).
  - Samples are generated within the feasible region.

Using a ``.csv`` file
---------------------

User needs to provide a valid path to read the ``.csv`` file.

.. code-block:: python

   model.dataset(parameters="path/to/parameters.csv")

- Number of columns in the CSV must match the number of parameters defined
- Order of columns should be consistent with the parameter definition

Box-Constrained Parameter Space
-------------------------------

User needs to provide both lower and upper bounds for all the parameters. User can provide a vector with different bounds for each parameter or a scalar as the same bound for all parameters.

.. code-block:: python

   model.box(lower=x_lower, upper=x_upper, num_samples = 500)

Defining a Simplex
------------------

A simplex is a :math:`d`-dimensional polytope defined by linear constraints of the form:

.. math::

   x \ge 0, \quad \mathbf{1}^\top x \le 1

More generally, a simplex can be represented as:

.. math::

   M x \le 1

.. code-block:: python

   model.simplex(M=M_matrix, num_samples=500)

Defining a Polytope Explicitly
------------------------------

Although the method is named ``simplex``, it supports general convex polytopes defined by linear inequalities. For example, to define the following polytope:

.. math::

   x_1 + x_2 + x_3 \le 1 \\
   x_1 + 2x_2 \le 0.8 \\
   0.5x_2 + x_3 \le 0.6 \\
   x_1 + 0.5x_3 \ge 0.2\\
   x_1 \ge 0, \quad x_2 \ge 0, \quad x_3 \ge 0

.. code-block:: python

   model.simplex(x.x1 >= 0.0, x.x2 >= 0.0, x.x3 >= 0.0,
                 x.x1 + x.x2 + x.x3 <= 1.0, x.x1 + 2.0 * x.x2 <= 0.8,
                 0.5 * x.x2 + x.x3 <= 0.6, x.x1 + 0.5 * x.x3 >= 0.2, num_samples=500)

The argument ``num_samples`` specifies the number of parameter samples to be generated from the defined polytope region. These sampled parameter points are used to construct the dataset for training and validation of the model.

- The total number of generated samples is equal to ``num_samples``
- The samples are split into training and validation sets based on ``train_frac``
- Each sample corresponds to one instance of the parametric optimization problem

If ``num_samples`` is not provided, a default value (typically ``1000``) is used.

Note
----

- Only one parameter space definition is required.
- If multiple are provided, the most recent call will overwrite the previous definition.
- The number of samples can be controlled via the configuration (e.g., ``num_samples``).

.. note::

   The user must define a parameter space for which the optimization problem is feasible.
   Currently, NLPOpt-Net does not automatically detect infeasibility for parameter realizations.
   Providing infeasible samples may lead to invalid training behavior or poor model performance.
