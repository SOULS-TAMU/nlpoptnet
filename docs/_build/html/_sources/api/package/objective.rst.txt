Objective
=========

The package supports both symbolic and callable objectives.

Symbolic objectives
-------------------

Typical symbolic forms include:

.. code-block:: python

   model.objective(0.5 * model.quad(model.Q, y) + model.lin(model.c, y))

.. code-block:: python

   model.objective(0.5 * model.quad(model.Q, y) + model.lin(model.c, model.sin(y)))

Callable objectives
-------------------

.. code-block:: python

   def objective(params, vars):
       y_value = vars["y"]
       return 0.5 * y_value @ model.Q @ y_value + model.c @ y_value

   model.objective(objective)

Important note
--------------

Callable objectives are flexible, but runs built from raw Python callables are not automatically reconstructible from saved metadata alone.

Relevant methods
----------------

The main user-facing helpers are:

- :meth:`nlpoptnet.api.NLPOptNet.objective`
- :meth:`nlpoptnet.api.NLPOptNet.lin`
- :meth:`nlpoptnet.api.NLPOptNet.batch_lin`
- :meth:`nlpoptnet.api.NLPOptNet.quad`
- :meth:`nlpoptnet.api.NLPOptNet.batch_quad`
- :meth:`nlpoptnet.api.NLPOptNet.batch_exp`
- :meth:`nlpoptnet.api.NLPOptNet.sin`
- :meth:`nlpoptnet.api.NLPOptNet.cos`
- :meth:`nlpoptnet.api.NLPOptNet.exp`
- :meth:`nlpoptnet.api.NLPOptNet.log`
- :meth:`nlpoptnet.api.NLPOptNet.sqrt`
- :meth:`nlpoptnet.api.NLPOptNet.abs`
