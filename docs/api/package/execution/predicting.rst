Predicting
==========

Predictions are produced with :meth:`nlpoptnet.api.NLPOptNet.predict`.

Examples
--------

Single sample:

.. code-block:: python

   prediction = model.predict([1.0, 2.0])

Batch:

.. code-block:: python

   batch_prediction = model.predict([[1.0, 2.0], [0.5, 1.5]])

Backend selection:

.. code-block:: python

   model.predict(values, projection_backend="auto")
   model.predict(values, projection_backend="jax")
   model.predict(values, projection_backend="native")

Runtime summary
---------------

The compact run summary is available through :meth:`nlpoptnet.api.NLPOptNet.summary`, and history plots can be created with :meth:`nlpoptnet.api.NLPOptNet.plot_history`.
