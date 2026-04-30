Object
======

The primary object in the package is :class:`nlpoptnet.api.NLPOptNet`.

For the configuration "tab" associated with this object, see :doc:`hyperparameters`.

Role of the object
------------------

``NLPOptNet`` is responsible for:

- storing symbolic parameters and variables,
- tracking constants and extracted problem data,
- recording objectives and constraints,
- loading or sampling parameter data,
- building the trainable model,
- training, saving, loading, and predicting.

Autodoc
-------

.. autoclass:: nlpoptnet.api.NLPOptNet
   :members:
   :special-members: __init__
   :member-order: bysource
