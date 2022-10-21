"""
feasel.parameter.params
=======================

This module combines all parameters from the *build*, *data*, *train* and
*callback* classes and provides them in a new grouped parameter class.

`feasel.parameter.ParamsLinear` only needs *build* and *data* parameters for an
instantiation, whereas `feasel.parameter.ParamsNN` also accepts the *train* and
*callback* parameters, that are typical for optimizers based on neural
networks.

"""

import json
from . import build, data, train, callback

class ParamsLinear:
  """
  Parameter container that stores all `build`, `data`, and `train` parameters
  and their methods, e.g. :func:`~.data..set_test_split()`.

  The build parameters are used for the network architecture. Train
  parameters are used to set all parameters for standard keras
  functionalities and data is used for the pre-processing of the data.

  This class is just a summary of all those adjustable parameter classes.

  Attributes
  ----------
  build
    The build parameters are set in :py:class:`feasel.parameter.build.Linear`.
  data : *class with parameters*
    The data parameters are set in :py:class:`feasel.parameter.data.Linear`.

  """
  def __init__(self):
    self.build = build.BuildParamsLinear()
    self.data = data.DataParamsLinear()
    self.params = {'build': self.build,
                   'data': self.data}

  def __repr__(self):
    return ('Parmeter container for FeaSel-Net linear transformations')

  def save(self, path):
    """
    Saves all parameters and stores them in the same directory as the keras
    model. It uses the HDF5 binary data format.

    Parameters
    ----------
    path : str
      The path where the params shall be stored.

    Returns
    -------
    None.

    """

    with open(path, 'w') as f:
      json.dump(self.params, f)

class ParamsNN(ParamsLinear):
  """
  Parameter container that stores all ``build``, ``callback``, ``data``, and
  ``train`` parameters and their methods, e.g. :func:`~.data.set_test_split()`.

  The ``build`` parameters are used for the neural network architecture.
  ``train`` parameters are used to set all parameters for standard keras
  functionalities and data is used for the pre-processing of the data.

  This class is just a summary of all these adjustable parameter classes.

  Attributes
  ----------
  build : class
    The build parameter container is defined in :class:`.build.NN`.
  callback : class
    The callback parameter container is defined in :class:`.callback.NN`.
  data : class
    The data parameter container is defined in :class:`.data.NN`.
  train : class
    The train parameter container is defined in :class:`.train.NN`.

  Methods
  -------
  save()
    Saves the current parameter configuration in a .json file.

  """
  def __init__(self):
    super().__init__()
    self.build = build.BuildParamsNN()
    self.data = data.DataParamsNN()
    self.train = train.TrainParamsNN()
    self.callback = callback.CallbackParamsNN()
    self.params = {'build': self.build,
                   'callback': self.callback,
                   'data': self.data,
                   'train': self.train}

  def __repr__(self):
    return ('Parmeter container for FeaSel-Net neural networks')