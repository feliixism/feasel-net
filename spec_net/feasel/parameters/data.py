from .base import BaseParams

_NORM_TYPES = {'min_max': ['min_max', 'min-max', 'min max', 'min-max scale'],
               'standardize': ['standardize', 'standard',
                               'z score', 'z-score']}

class DataParams(BaseParams):
  def __init__(self,
               normalization=None,
               sample_axis=None,
               input_layer=None):
    """


    Parameters
    ----------
    normalization : str, optional
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
    self.normalization = normalization
    self.sample_axis = sample_axis
    self.input_layer = input_layer

    self._MAP = {'normalization': self.set_normalization,
                 'sample_axis': self.set_sample_axis,
                 'input_layer': self.set_input_layer}

  def __repr__(self):
    return ('Parmeter container for generic data processing\n'
            f'{self.__dict__}')

  # SETTERS:
  def set_normalization(self, normalization):
    if normalization:
      self.normalization = self._get_normalization(normalization)
    else:
      self.normalization = None

  def set_sample_axis(self, sample_axis):
    self.sample_axis = sample_axis

  def set_input_layer(self, input_layer):
    self.input_layer = input_layer

  # HELPERS:
  def _get_normalization(self, normalization):
    """
    Provides different aliases for all possible noramlizations and searches for
    the corresponding normalization (e.g. 'min max' --> 'min_max').

    Parameters
    ----------
    normalization : str
      Normalization alias.

    Raises
    ------
    NameError
      If normalization is not a valid alias.

    Returns
    -------
    NORM : str
      The proper normalization used for the following operations inside the
      class.

    """
    for NORM in _NORM_TYPES:
      if normalization in _NORM_TYPES[f'{NORM}']:
        return NORM
    raise NameError(f"'{normalization}' is not a valid normalization.")