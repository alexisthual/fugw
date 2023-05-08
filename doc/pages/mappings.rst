--------------------
:mod:`fugw.mappings`
--------------------

Most likely, this is the module you want to use.
It comprises 2 main classes: :class:`fugw.mappings.FUGW`
and :class:`fugw.mappings.FUGWSparse`,
each of which comes with ``.fit()``, ``transform()`` and ``.inverse_transform()``
methods to easily fit your FUGW models and apply them to new data.

.. warning::

   Mappings overwrite ``__getstate__()`` and ``__setstate__()`` methods
   in order to store the model hyper-parameters and weights separately in the same
   pickle files. This allows for loading hyper-parameters only, which results in
   much faster loading times.
   
   However, this means you should be careful when
   saving or loading mappings with pickle. If you want to save/load a mapping, use
   ``fugw.utils.save()`` and ``fugw.utils.load()`` methods instead.


.. currentmodule:: fugw.mappings

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FUGW
   FUGWSparse
