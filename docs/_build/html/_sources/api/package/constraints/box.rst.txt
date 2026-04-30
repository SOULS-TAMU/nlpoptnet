Box
===

Box constraints are kept on a dedicated path instead of being merged into the general inequality residuals.

Affine vector bounds
--------------------

.. code-block:: python

   model.constraints.box.add(
       var=y,
       lower=model.l + model.lin(model.L, x),
       upper=model.u + model.lin(model.U, x),
   )

Scalar bounds
-------------

.. code-block:: python

   model.constraints.box.add(y.y1 >= 0.0, y.y2 <= 1.0)

Why this matters
----------------

The internal implementation converts box bounds into affine lower and upper bound operators, which are then enforced directly inside the projection layer.
