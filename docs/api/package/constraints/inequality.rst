Inequality
==========

Inequality constraints are added through ``model.constraints.inequality.add(...)``.

Examples
--------

Basic nonlinear inequality:

.. code-block:: python

   model.constraints.inequality.add(
       y.y1**2 + y.y3**2 <= 2.0,
   )

QP form:

.. code-block:: python

   model.constraints.inequality.add(
       model.lin(model.C, y) <= model.d + model.lin(model.D, x),
   )

QCQP form:

.. code-block:: python

   model.constraints.inequality.add(
       model.batch_quad(model.C, y) + model.batch_lin(model.d, y)
       <= model.e + model.batch_lin(model.E, x),
   )

NLP form:

.. code-block:: python

   model.constraints.inequality.add(
       model.batch_lin(model.a, model.batch_exp(y)) + model.batch_quad(model.W, y)
       <= model.beta + model.batch_lin(model.E, x),
   )
