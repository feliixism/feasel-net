from .build import BuildParams
from .callback import CallbackParams
from .data import DataParams
from .train import TrainParams

class Params:
  def __init__(self):
    """
    Parameter container that stores all 'build', 'tain' and 'data' parameter
    and their methods, e.g. 'train.set_learning_rate()'.

    The build parameters are used for the network architecture. Train
    parameters are used to set all parameters for standard keras
    functionalities and data is used for the pre-processing of the data.

    This class is just a summary of all those adjustable parameter classes.

    Returns
    -------
    None.

    """
    self.build = BuildParams()
    self.callback = CallbackParams()
    self.data = DataParams()
    self.train = TrainParams()

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
    return # has to be implemented


