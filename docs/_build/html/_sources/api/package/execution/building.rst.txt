Building
========

The build stage is triggered with :meth:`nlpoptnet.api.NLPOptNet.build`.

What build does
---------------

- validates the model definition,
- normalizes the configuration,
- constructs the symbolic ``jaxmodel`` problem,
- prepares bound operators,
- loads or samples parameter data,
- splits train and validation data,
- prepares fixed batches,
- initializes the trainable backbone and projection pipeline,
- optionally runs a JIT warmup pass.

Typical use
-----------

.. code-block:: python

   model.dataset(parameters="notebooks/data/qp/parameters.csv")
   model.build()
