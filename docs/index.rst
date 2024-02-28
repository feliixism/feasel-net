FeaSel-Net
==========
**FeaSel-Net** is a python package that enables feature selection algorithms embedded in a neural network architecture. It combines a leave-one-out cross-validation (LOOCV) type of feature selection algorithm with recursive pruning of the input nodes, such that only the most relevant nodes with the richest information are kept for the subsequent optimization task. The recursive pruning is undertaken by employing a Feature Selection Callback at certain points of the optimization process. The precise procedure is explained in Sequence of Events. Originally developed for serving the task of finding biomarkers in biological tissues, the algorithm is generically coded such that it is able to select features for all kinds of classification tasks.

The package is an extension for the keras and tensorflow libraries. Please see the links for further information on their software packages and to get a grasp of neural networks in general and the constructs used for FeaSel-Net.

Contribute
----------
- Issue Tracker: https://github.tik.uni-stuttgart.de/FelixFischer/FeaSel-Net/issues

- Source Code: https://github.tik.uni-stuttgart.de/FelixFischer/FeaSel-Net

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 6

   information/about
   information/install
   information/SOE
   information/quickstart
   module/modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
