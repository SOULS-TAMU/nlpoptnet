Defining the Problem
====================

This notebook provides the way of defining the model in NLPOpt-Net. You can define block of constraints when you have exploitable structure and also can define explicit constraitns. Along with objective and constraints this notebook provides the method of defining variable bounds that can be parameter independent or parameter dependent. We consider the general form of the problem as follows:

.. math::

   \begin{aligned}
   \min_y &\,\, f(x,y)\\
   \textrm{s.t.} &\, \, g(x,y) \leq 0 \\
   &\, \, h(x,y) = 0 \\
   &\, \, l(x) \leq y \leq u(x)
   \end{aligned}

The general workflow for defining a model in NLPOpt-Net is as follows:

- Configure training settings
- Create a ``NLPOptNet`` object
- Define the variables and parameters
- Define the objective, constraints, variable bounds
- Define parameter space
- Build the model
- Train and infer

Creating the Model Object
-------------------------

Ensure that all required configuration fields are provided. Optional parameters can be specified based on user preference. In this example we only provide the required configuration fields.

.. code-block:: python

   from nlpoptnet import NLPOptNet

   CONFIG = {
       'epochs': 1000,
       'batch_size': 40,
       'learning_rate': 1e-3,
       'hidden_size': 64,
       'hidden_layers': 2
   }

   model = NLPOptNet(config=CONFIG, type=None, name='Example_Model')

Defining the Variables and Parameters
-------------------------------------

You can define the parameter and variable names explicitly or by providing a list of strings.

.. code-block:: text

   parameter_set = model.add_parameter([parameter_1, parameter_2, ...])

- ``parameter_set``: handle to the defined parameters
- ``parameter_i``: user-defined string identifiers (e.g., ``"temperature"``, ``"pressure"``, ``"x1"``,...)

.. code-block:: text

   variable_set = model.add_variable([variable_1, variable_2, ...])

- ``variable_set``: handle to the defined variables
- ``variable_i``: user-defined string identifiers (e.g., ``"flow"``, ``"y1"``, ...)

It is convenient to define explicitely when system has few parameters and variables.

.. code-block:: python

   x = model.add_parameter(['x1', 'x2'])
   y = model.add_variable(['y1', 'y2', 'y3', 'y4'])

For a large number of parameters and variables you may consider the following:

.. code-block:: python

   no_of_parameters = 50
   no_of_variables = 100
   parameter_names = [f'x{i+1}' for i in range(no_of_parameters)]
   variable_names = [f'y{i+1}' for i in range(no_of_variables)]
   x = model.add_parameter(parameter_names)
   y = model.add_variable(variable_names)

Extracting Model Constants from Previously Saved Files
------------------------------------------------------

Currently we support loading arrays from a ``.npz`` file. The model will automatically resgister the contained arrays as model constants.

.. code-block:: python

   model.extract(DATA_DIRECTORY / 'problem.npz')

This will load all arrays stored in ``problem.npz``. The following is an example of the output:

Loaded problem constants: ['A', 'B', 'C', 'L', 'Q', 'U', 'b', 'c', 'd', 'l', 'u']

User can use them by calling ``model.A``, ``model.B``, ...

Defining Model Constants Directly
---------------------------------

Model constants (arrays) can be added directly using the provided APIs. The following examples illustrate how to define matrices, vectors, and tensors:

.. code-block:: python

   # Matrix (2D array)
   model.Q = model.matrix([[1.0, 0.0], [0.0, 1.0]])
   model.A = model.matrix([[1.0, 0.0], [0.0, 1.0]])

   # Vector (1D array)
   model.c = model.vector([1.0, 2.0])

   # Tensor (multi-dimensional array)
   model.T = model.tensor([
       [[1.0, 0.0], [0.0, 1.0]],
       [[2.0, 0.0], [0.0, 2.0]]
   ])

Model constants can be accessed directly as attributes (e.g., ``model.Q``). To view their values, use ``model.Q.value`` or convert to a NumPy array using ``np.array(model.Q)``.

Defining the Objective
----------------------

The objective can be defined using built-in structured functions or explicitly using expressions. NLPOpt-Net provides several built-in functions to construct structured objectives:

.. code-block:: python

   # Matrix operations
   model.lin(model.A, y)        # Linear: A @ y
   model.quad(model.Q, y)       # Quadratic: y^T Q y
   model.batch_lin(model.A, y)  # Batched linear mapping
   model.batch_quad(model.Q, y) # Batched quadratic form

   # Elementwise nonlinear functions
   model.sin(expr)        # Sine
   model.cos(expr)        # Cosine
   model.exp(expr)        # Exponential
   model.log(expr)        # Logarithm
   model.sqrt(expr)       # Square root
   model.abs(expr)        # Absolute value

.. code-block:: python

   # Quadratic objective function
   model.objective(0.5 * model.quad(model.Q, y) + model.lin(model.c, y))

.. code-block:: python

   # Nonlinear bbjective function
   model.objective(0.5 * model.quad(model.Q, y)          # 0.5 y^T Q y
                   + model.lin(model.c, y)               # c^T y
                   + model.lin(model.p, model.exp(y))    # p^T exp(y)
                   )

.. code-block:: python

   # Explicit expression
   model.objective(0.5 * (y[0]*y[0] + y[1]*y[1]) + 2*y[0] + 3*y[1] + 1*model.exp(y[0]) + 2*model.exp(y[1]))

Note that parameters can also be included in the objective in similar way.

Defining Constraints
--------------------

Similar to the objective, constraints can be defined using structured functions or explicit expressions.

.. code-block:: python

   # Linear constraints
   model.constraints.equality.add(model.lin(model.A, y) == model.b + model.lin(model.B, x))

   model.constraints.inequality.add(model.lin(model.C, y) <= model.d + model.lin(model.D, x))

.. code-block:: python

   # Quadratic constraints
   model.constraints.inequality.add(model.batch_quad(model.C, y) + model.batch_lin(model.d, y) <= model.e + model.batch_lin(model.E, x))

.. code-block:: python

   # Nonlinear constraints
   model.constraints.inequality.add(model.batch_quad(model.C, y) + model.batch_lin(model.d, y) <= model.e + model.batch_lin(model.E, x))

.. code-block:: python

   # Explicit expression as constraints
   model.constraints.equality.add(y.y1 + y.y2 == x.x1,
                                  y.y2 - y.y3 == x.x2)

   model.constraints.inequality.add(y.y1 ** 2 + y.y3 ** 2 <= 2.0,
                                    y.y1 + 0.5 * y.y2 <= 1.5)

Defining Box/Bounding Constraints
---------------------------------

Box constraints define lower and upper bounds on the decision variables. Bounds may be parameterized or some decision variable may be unbounded.

.. code-block:: python

   # Parameterized bounds
   model.constraints.box.add(var=y,
                             lower=model.l + model.lin(model.L, x),
                             upper=model.u + model.lin(model.U, x))

.. code-block:: python

   # Non-parameterized bounds
   model.constraints.box.add(var=y, lower=model.l, upper=model.u)

If only a lower or upper bound is available, provide only that bound. If a variable has no lower or upper bound, leave it unbounded by not specifying a bound for that variable. For vector bounds, use ``None``, ``-np.inf``, or ``np.inf`` depending on how the bound data is represented. If the variables are unbounded do not call ``.constraints.box.add``.

.. code-block:: python

   # Lower bound only
   model.constraints.box.add(var=y, lower=model.l)

   # Upper bound only
   model.constraints.box.add(var=y,upper=model.u)

   # Example: y[0] has bounds, y[1] is unbounded
   l = np.array([0.0, -np.inf])
   u = np.array([1.0,  np.inf])

After defining the objective, constraints, and bounds, the next step is to specify the parameter space.
