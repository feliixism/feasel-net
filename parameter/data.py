"""
feasel.parameter.data
=====================
"""

from .base import BaseParams

_NORM_TYPES = {'min_max': ['min_max', 'min-max',
                           'min max', 'min-max scale'],
               'standardize': ['standardize', 'standard',
                               'z score', 'z-score']}

class DataParamsLinear(BaseParams):
  """
  Parameter class for all data parameters.

  Attributes
  ----------
  data_norm : str, optional
    The normalization type of the input data. The default is None.
  sample_axis : int, optional
    The axis of the input array X where the samples are located. The default
    is None.
  mean_centered : bool,
    Defines whether or not the data is mean-centered. The default is 'False'.
  test_split : float,
    The ratio for the test data. The default is '0.2'.

  """
  def __init__(self,
               data_norm=None,
               sample_axis=None,
               mean_centered=False,
               test_split=0.2):

    super().__init__()
    self.data_norm = data_norm
    self.sample_axis = sample_axis
    self.mean_centered = mean_centered
    self.test_split = test_split

    self._MAP = {'data_norm': self.data_norm,
                 'sample_axis': self.set_sample_axis,
                 'mean_centered': self.set_mean_centered}

  def __repr__(self):
    return ('Parmeter container for generic data processing\n'
            f'{self.__dict__}')

  # SETTERS:
  def set_data_norm(self, data_norm):
    self.data_norm = self._get_data_norm(data_norm)

  def set_sample_axis(self, sample_axis):
    self.sample_axis = sample_axis

  def set_mean_centered(self, mean_centered):
    self.mean_centered = mean_centered

  def set_test_split(self, test_split):
    self.test_split = test_split

  # HELPERS:
  def _get_data_norm(self, data_norm):
    """
    Provides different aliases for all possible normalizations and searches for
    the corresponding normalization (e.g. 'min max' --> 'min_max').

    Parameters
    ----------
    data_norm : str
      Normalization alias.

    Raises
    ------
    NameError
      If data_norm is not a valid alias.

    Returns
    -------
    NORM : str
      The proper normalization used for the following operations inside the
      class.

    """
    if not data_norm:
      return None
    for NORM in _NORM_TYPES:
      if data_norm in _NORM_TYPES[f'{NORM}']:
        return NORM
    raise NameError(f"'{data_norm}' is not a valid normalization.")

class DataParamsNN(BaseParams):
  def __init__(self,
               data_norm=None,
               sample_axis=None,
               input_layer=None):
    """


    Parameters
    ----------
    data_norm : str, optional
      The normalization type of the input data. The default is None.
    sample_axis : int, optional
      The axis of the input array X where the samples are located. The default
      is None.
    input_layer : str, optional
      The name of the input layer. The default is None.

    Returns
    -------
    None.

    """
    super().__init__()
    self.data_norm = data_norm
    self.sample_axis = sample_axis
    self.input_layer = input_layer

    self._MAP = {'data_norm': self.set_data_norm,
                 'sample_axis': self.set_sample_axis,
                 'input_layer': self.set_input_layer}

  def __repr__(self):
    return ('Parmeter container for generic data processing\n'
            f'{self.__dict__}')

  # SETTERS:
  def set_data_norm(self, data_norm):
    if data_norm:
      self.data_norm = self._get_data_norm(data_norm)
    else:
      self.data_norm = None

  def set_sample_axis(self, sample_axis):
    self.sample_axis = sample_axis

  def set_input_layer(self, input_layer):
    self.input_layer = input_layer

  # HELPERS:
  def _get_data_norm(self, data_norm):
    """
    Provides different aliases for all possible noramlizations and searches for
    the corresponding normalization (e.g. 'min max' --> 'min_max').

    Parameters
    ----------
    data_norm : str
      Normalization alias.

    Raises
    ------
    NameError
      If data_norm is not a valid alias.

    Returns
    -------
    NORM : str
      The proper normalization used for the following operations inside the
      class.

    """
    for NORM in _NORM_TYPES:
      if data_norm in _NORM_TYPES[f'{NORM}']:
        return NORM
    raise NameError(f"'{data_norm}' is not a valid normalization.")
