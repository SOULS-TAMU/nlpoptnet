Loading
=======

Saved runs can be restored with :meth:`nlpoptnet.api.NLPOptNet.load`.

Example
-------

.. code-block:: python

   restored = NLPOptNet()
   restored.load("demo_qp_20260416_120000/metadata.json")

Limitations
-----------

Automatic reload is supported for serializable symbolic problems. Runs that depend on raw Python objective callables or callable constraint blocks are intentionally rejected during ``load(...)`` because the original Python code is not reconstructed from metadata.
