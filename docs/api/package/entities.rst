Entities
========

In the package workflow, entities are the symbols and constants that appear in a problem definition.

Parameters and variables
------------------------

- Parameters are registered with :meth:`nlpoptnet.api.NLPOptNet.add_parameter`.
- Variables are registered with :meth:`nlpoptnet.api.NLPOptNet.add_variable`.
- Access is expression-based, for example ``x.x1`` or ``y.y2``.

Constants
---------

Constants can be introduced manually with:

- :meth:`nlpoptnet.api.NLPOptNet.matrix`
- :meth:`nlpoptnet.api.NLPOptNet.vector`
- :meth:`nlpoptnet.api.NLPOptNet.tensor`

or loaded in bulk with :meth:`nlpoptnet.api.NLPOptNet.extract`.

Autodoc
-------

.. autoclass:: nlpoptnet.api.Constant
   :members:
   :member-order: bysource

.. autoclass:: nlpoptnet.api.Expression
   :members:
   :member-order: bysource

.. autoclass:: nlpoptnet.api.VectorExpression
   :members:
   :member-order: bysource
