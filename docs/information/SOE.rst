Sequence of Events
==================

1. Initiallizing Neural Network
-------------------------------

The first step of the algorithm can be thought of a simple optimization problem initiallized with the inputs and a binary mask for those inputs with only ones as its entries. This behaviour is induced by using a newly created layer type called LinearPass.

2. Training until trigger conditions are met
--------------------------------------------

The neural network optimizes the classification results until one of the following options happen:

- the training (or validation) loss value is beneath a certain threshold

- the training (or validation) accuracy value is above a certain threshold

Then - for the sake of consistency - it will count how many times in a row the conditions are met. If this happens for multiple epochs, the actual pruning event will start that consists of estimating the importance and eliminating uninformative features.

3. Importance estimation
------------------------

As soon as the callback is triggered, the evaluation of the features is done.