NonLinear Programs (NLPs)
=========================

Consider the following QCQP problem:

.. math::

   \begin{aligned}
   \min_{y} \quad & \frac{1}{2} y^\top Q y + c^\top y \\
   \text{s.t.} \quad
   & A y = b + B x, \\
   & a_i^\top \exp(y) + y^\top W y \leq \beta_i + E_i x, \quad i = 1, 2, \ldots, m_i, \\
   & l + L x \leq y \leq u + U x.
   \end{aligned}

where, :math:`Q \in \mathbb{R}^{n \times n} \succeq 0`, :math:`c \in \mathbb{R}^n`, :math:`A \in \mathbb{R}^{n_{\textrm{eq}} \times n}`, :math:`b \in \mathbb{R}^{n_{\textrm{eq}}}`, :math:`B \in \mathbb{R}^{n_{\textrm{eq}} \times p}`, :math:`a_i \in \mathbb{R}^n`, :math:`W \in \mathbb{R}^{n \times n} \succeq 0`, :math:`\beta_i \in \mathbb{R}`, :math:`E_i \in \mathbb{R}^{1 \times p}` for :math:`i = 1,2,\ldots,n_{\textrm{ineq}}`, :math:`l \in \mathbb{R}^n`, :math:`L \in \mathbb{R}^{n \times p}`, :math:`u \in \mathbb{R}^n`, and :math:`U \in \mathbb{R}^{n \times p}`. The parameter :math:`x \in \mathbb{R}^p` varies across problem instances. We aim to learn to approximate the optimal solution :math:`y \in \mathbb{R}^n` given a parameter realization :math:`x`. For this example consider, :math:`n=10, p=5,n_\textrm{eq}=5,n_\textrm{ineq}=5`.

Import Libraries
----------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from nlpoptnet import NLPOptNet

Set configuration and Create the model object
---------------------------------------------

.. code-block:: python

   CONFIG = {
       'epochs': 100,
       'batch_size': 32,
       'learning_rate': 1e-3,
       'train_frac': 0.8,
       'hidden_size': 64,
       'hidden_layers': 2,
       'seed': 42,
       'alpha_consistency': 10.0,
       'cp_mode': 'fixed',
       'cp_iters': 300,
       'cp_tol': 1e-9,
       'safety': 0.95,
       'knorm_iters': 15,
       'knorm_seed': 42,
       'adjoint_iters': 20,
       'k_layer': 3,
       'use_ruiz': True,
       'ruiz_iters': 5,
       'dtype': 'float64',
       'print_every': 10,
       'device': 'auto',
       'verbose': True,
   }

   model = NLPOptNet(config=CONFIG, type='nlp', name='Example_NLP')
   x = model.add_parameter([f'x{i+1}' for i in range(5)])
   y = model.add_variable([f'y{i+1}' for i in range(10)])

Define the problem constants
----------------------------

.. code-block:: python

   Q = np.array([[2.1, 0.1, 0.0, 0.05, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0],
                 [0.1, 2.0, 0.08, 0.0, 0.03, 0.0, 0.02, 0.0, 0.0, 0.0],
                 [0.0, 0.08, 1.9, 0.07, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
                 [0.05, 0.0, 0.07, 2.2, 0.06, 0.0, 0.0, 0.0, 0.02, 0.0],
                 [0.0, 0.03, 0.0, 0.06, 2.05, 0.0, 0.0, 0.0, 0.0, 0.02],
                 [0.02, 0.0, 0.0, 0.0, 0.0, 1.8, 0.05, 0.0, 0.0, 0.0],
                 [0.0, 0.02, 0.0, 0.0, 0.0, 0.05, 1.85, 0.04, 0.0, 0.0],
                 [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.04, 1.95, 0.03, 0.0],
                 [0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03, 1.75, 0.04],
                 [0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.04, 1.88]], dtype=float)

   c = np.array([0.65, 0.75, 0.85, 0.95, 1.05, 0.55, 0.7, 0.9, 1.1, 1.2], dtype=float)

   A = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.0, 0.05, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.15, -0.05, 0.0, 0.1],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -0.1, 0.0, 0.1, 0.05, 0.0],
                 [0.0, 0.0, 0.0, 1.0, 0.0, 0.05, 0.0, 0.0, -0.1, 0.15],
                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.05, 0.1, 0.0, -0.1]], dtype=float)

   b = np.array([0.28, -0.1915, 0.083, -0.0175, 0.1455], dtype=float)

   B = np.array([[0.0905, 0.027, -0.01, 0.0035, 0.0275],
                 [-0.02, 0.073, 0.0235, -0.0095, -0.001],
                 [0.0145, -0.028, 0.087, 0.0155, -0.0175],
                 [-0.0075, 0.0235, -0.022, 0.078, 0.0005],
                 [0.032, 0.0035, 0.0055, -0.02, 0.083]], dtype=float)

   a_nlp = np.array([[0.08, 0.0, 0.04, 0.0, 0.0, 0.03, 0.0, 0.0, 0.02, 0.0],
                     [0.0, 0.07, 0.0, 0.05, 0.0, 0.0, 0.02, 0.0, 0.0, 0.03],
                     [0.03, 0.0, 0.06, 0.0, 0.04, 0.0, 0.0, 0.02, 0.0, 0.0],
                     [0.0, 0.02, 0.0, 0.08, 0.0, 0.05, 0.0, 0.0, 0.03, 0.0],
                     [0.04, 0.0, 0.0, 0.0, 0.07, 0.0, 0.03, 0.0, 0.0, 0.02]], dtype=float)

   W_nlp = np.array([[[0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

                     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.035, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

                     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.035, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

                     [[0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02]],

                     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.035]]], dtype=float)

   beta = np.array([0.85662226, 0.77398075, 0.81796323, 0.82510558, 0.82826729], dtype=float)

   E = np.array([[0.08, -0.04, 0.0, 0.024, 0.016],
                 [-0.016, 0.064, 0.032, 0.0, -0.008],
                 [0.0, 0.024, -0.056, 0.04, 0.016],
                 [0.032, 0.0, 0.016, -0.048, 0.024],
                 [-0.024, 0.016, 0.0, 0.032, -0.064]], dtype=float)

   l = np.array([-1.05, -1.4, -1.15, -1.3, -1.13, -0.95, -1.5, -1.07, -1.35, -1.2], dtype=float)

   L = np.array([[0.1, 0.02, -0.01, 0.0, 0.03],
                 [-0.02, 0.08, 0.02, -0.01, 0.0],
                 [0.01, -0.03, 0.09, 0.02, -0.02],
                 [0.0, 0.02, -0.02, 0.07, 0.01],
                 [0.03, 0.0, 0.01, -0.02, 0.08],
                 [-0.04, 0.01, 0.0, 0.03, -0.01],
                 [0.02, -0.05, 0.01, 0.0, 0.02],
                 [0.0, 0.03, -0.04, 0.01, 0.0],
                 [0.01, 0.0, 0.02, -0.05, 0.03],
                 [-0.03, 0.02, 0.0, 0.01, -0.04]], dtype=float)

   u = np.array([1.45, 1.1, 1.35, 1.2, 1.37, 1.55, 1.0, 1.43, 1.15, 1.3], dtype=float)

   U = np.array([[0.1, 0.02, -0.01, 0.0, 0.03],
                 [-0.02, 0.08, 0.02, -0.01, 0.0],
                 [0.01, -0.03, 0.09, 0.02, -0.02],
                 [0.0, 0.02, -0.02, 0.07, 0.01],
                 [0.03, 0.0, 0.01, -0.02, 0.08],
                 [-0.04, 0.01, 0.0, 0.03, -0.01],
                 [0.02, -0.05, 0.01, 0.0, 0.02],
                 [0.0, 0.03, -0.04, 0.01, 0.0],
                 [0.01, 0.0, 0.02, -0.05, 0.03],
                 [-0.03, 0.02, 0.0, 0.01, -0.04]], dtype=float)

   model.Q = model.matrix(Q)
   model.c = model.vector(c)
   model.A = model.matrix(A)
   model.b = model.vector(b)
   model.B = model.matrix(B)
   model.a_nlp = model.matrix(a_nlp)
   model.W_nlp = model.tensor(W_nlp)
   model.beta = model.vector(beta)
   model.E = model.matrix(E)
   model.l = model.vector(l)
   model.L = model.matrix(L)
   model.u = model.vector(u)
   model.U = model.matrix(U)

Define the objective, constraints, and the parameter sample space
-----------------------------------------------------------------

.. code-block:: python

   model.objective(0.5 * model.quad(model.Q, y) + model.lin(model.c, y))
   model.constraints.equality.add(model.lin(model.A, y) == model.b + model.lin(model.B, x))
   model.constraints.inequality.add(model.batch_lin(model.a_nlp, model.exp(y)) + model.batch_quad(model.W_nlp, y) <= model.beta + model.batch_lin(model.E, x))
   model.constraints.box.add(var=y, lower=model.l + model.lin(model.L, x), upper=model.u + model.lin(model.U, x))

   model.box(lower=np.array([-1.0]*5), upper=np.array([1.0]*5), num_samples=200)

Build, train and use the model
------------------------------

.. code-block:: python

   model.build()
   result = model.optimize()
   run_dir = result['output_dir']
   print('run_dir =', run_dir)

   model.summary()
   model.plot_history()

   X_test = np.array([
       [ 0.00,  0.00,  0.00,  0.00,  0.00],
       [ 0.50, -0.25,  0.75, -0.50,  0.25],
       [-0.80,  0.40, -0.20,  0.60, -0.10],
   ], dtype=float)

   pred = model.predict(X_test, projection_backend='jax')
   print('predicted y shape:', np.asarray(pred).shape)
   pd.DataFrame(np.asarray(pred), columns=[f'y{i+1}' for i in range(10)])
