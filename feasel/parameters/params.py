from . import build, data, train, callback

import json

class ParamsLinear:
  """
  Parameter container that stores all `build`, `data`, and `train` parameters
  and their methods, e.g. :func:`~.data..set_test_split()`.

  The build parameters are used for the network architecture. Train
  parameters are used to set all parameters for standard keras
  functionalities and data is used for the pre-processing of the data.

  This class is just a summary of all those adjustable parameter classes.

  Methods
  -------
  save()
    Saves the current parameters in a json file.

  """
  def __init__(self):
    self.build = build.Linear()
    self.data = data.Linear()
    self.params = {'build': self.build, 'data': self.data}

  def __repr__(self):
    return ('Parmeter container for the summary of all parameter classes\n'
            f'{self.__dict__}')

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
  Parameter container that stores all `build`, `callback`, `data`, and `train`
  parameters and their methods, e.g. :func:`~.data..set_test_split()`.

  The build parameters are used for the network architecture. Train
  parameters are used to set all parameters for standard keras
  functionalities and data is used for the pre-processing of the data.

  This class is just a summary of all those adjustable parameter classes.

  Methods
  -------
  save()
    Saves the current parameters in a json file.

  """
  def __init__(self):
    super().__init__()
    self.build = build.NN()
    self.data = data.NN()
    self.train = train.NN()
    self.callback = callback.NN()
    self.params = {'build': self.build, 'callback': self.callback,
                   'data': self.data, 'train': self.train}