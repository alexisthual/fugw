.. _api_references:

==============
API references
==============

This package consists of two main modules: ``fugw.mappings`` and ``fugw.solvers``.

The former contains two classes to easily handle mappings (also called transport plans or alignments). These classes come with handy methods such as ``.fit()`` to compute a mapping between two distributions or ``.transform()`` to transport signal from one domain to the other.
For computing mappings, they both make use of the ``fugw.solvers`` module, which computes solutions for the underlying optimization problem.

Besides, ``fugw.utils`` module contains some utility functions, for saving and loading mappings for instance.

.. toctree::
   :hidden:

   ./mappings.rst
   ./solvers.rst
   ./utils.rst
