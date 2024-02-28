FeaSel-Net
==========

*FeaSel* is the base module for the FeaSel-Net algorithm. It provides different
submodules to automatically and generically build fully connected deep neural
networks (FCDNNs) with reasonable tested hyperparameters to start with. It also
provides two different possibilities to select features from a bigger subset of
input features:

1. Linear Feature Selection
---------------------------

The linear features selection (FS) is done using PCAs and LDAs depending on the
type of learning task. If the data shall be clustered without any knowledge on
the categorization of it, PCA will do the job. LDA is used, when the labels are
known already.
The features can be selected by evaluating the loadings for each
transformation, which are treated as equivalents to weights and thus
information content.

Please have a look at :mod:`feasel.linear_transformation` for further
information.

2. Non-linear Feature Selection
-------------------------------

The non-linear feature selection is done by recursively pruning inputs in a
neural network environment.