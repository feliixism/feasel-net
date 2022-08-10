from .base import BaseParams

_NORM_TYPES = {'min_max': ['min_max', 'min-max', 'min max', 'min-max scale'],
               'standardize': ['standardize', 'standard',
                               'z score', 'z-score'],
               'None': [None]}

class DataParams(BaseParams):
  def __init__(self,
               normalization=None,
               sample_axis=None,
               mean_centered=False,
               test_split=0.2):
    """
    Parameter class for all data parameters.

    Parameters
    ----------
    normalization : str, optional
      The normalization type of the input data. The default is None.
    sample_axis : int, optional
      The axis of the input array X where the samples are located. The default
      is None.
    mean_centered : bool,
      Defines whether or not the data is mean-centered. The default is 'False'.
    test_split : float,
      The ratio for the test data. The default is '0.2'.

    Returns
    -------
    None.

    """
    super().__init__()
    self.normalization = normalization
    self.sample_axis = sample_axis
    self.mean_centered = mean_centered
    self.test_split = test_split

    self._MAP = {'normalization': self.set_normalization,
                 'sample_axis': self.set_sample_axis,
                 'mean_centered': self.set_mean_centered}

  def __repr__(self):
    return ('Parmeter container for generic data processing\n'
            f'{self.__dict__}')

  # SETTERS:
  def set_normalization(self, normalization):
    self.normalization = self._get_normalization(normalization)

  def set_sample_axis(self, sample_axis):
    self.sample_axis = sample_axis

  def set_mean_centered(self, mean_centered):
    self.mean_centered = mean_centered

  def set_test_split(self, test_split):
    self.test_split = test_split

  # HELPERS:
  def _get_normalization(self, normalization):
    """
    Provides different aliases for all possible normalizations and searches for
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
