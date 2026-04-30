Equality
========

Equality constraints are added through ``model.constraints.equality.add(...)``.

Examples
--------

Expression-based:

.. code-block:: python

   model.constraints.equality.add(
       y.y1 + y.y2 - x.x1 == 0,
       y.y2 - y.y3 - x.x2 == 0,
   )

Structured affine:

.. code-block:: python

   model.constraints.equality.add(
       model.lin(model.A, y) == model.b + model.lin(model.B, x),
   )

Callable block:

.. code-block:: python

   def equality_block(params, vars):
       x_value = params["x"]
       y_value = vars["y"]
       return y_value[:2] - x_value

   model.constraints.equality.add(equality_block)
