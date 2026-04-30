Training
========

The training stage is triggered with :meth:`nlpoptnet.api.NLPOptNet.optimize`.

Returned value
--------------

.. code-block:: python

   result = model.optimize()

   print(result["output_dir"])
   print(result["metadata_path"])

The returned dictionary contains:

- ``output_dir``
- ``metadata_path``
- ``summary``
- ``history``

Saved artifacts
---------------

Training writes a timestamped run directory containing:

- ``metadata.json``
- ``summary.json``
- ``history.csv``
- ``parameters.csv``
- ``predicted_variables.csv``
- ``model_weights.npz``
- ``backbone_weights.npz``
- ``problem_constants.npz``
- ``problem.npz`` when the problem was extracted from disk
- ``projection_native.json``
