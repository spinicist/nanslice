NANSlice - Not ANother Slice tool
====================================

(Alternative names are Not-A-Number Slice tool & Neuroimaging ANalysis Slice tool)

This is a pure Python module for taking slices through MR Images and displaying
them in beautiful ways. It is friendly to both clinical and pre-clinical data,
and includes dual-coding (http://dx.doi.org/10.1016/j.neuron.2012.05.001) overlays.

This is an example of an image you can make with it:

.. image:: doc/screenshot.png

Along with the :doc:`doc/nanslice` that can be used in your Python scripts, there are
several utitlity functions for use in :py:mod:`~nanslice.jupyter` notebooks, including a three-plane
viewer. There are also three command line tools that will installed to your $PATH:

- :py:mod:`~nanslice.nanslicer` Produces different kind of slice-plots including color bars
- :py:mod:`~nanslice.nanviewer` A three-plane viewer. Requires PyQt.
- :py:mod:`~nanslice.nanvideo` Converts time-series images to a movie file for easy viewing.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   doc/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Legal
=====

Copyright 2018 tobias.wood@kcl.ac.uk

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
