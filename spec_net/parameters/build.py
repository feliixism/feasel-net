from .base import BaseParams

_ARCHITECTURE_TYPES = {'exponential': ['exp-down', 'exponential', 'exp.',
                                       'exp down', 'exponential decrease'],
                       'linear': ['down', 'linear', 'linear decrease'],
                       'const': ['const', 'constant']}

class BuildParams(BaseParams):
  def __init__(self,
               architecture_type='down',
               n_layers=3,
               n_features=None,
               n_nodes=None):
    """
    Parameter class for all build parameters.

    Parameters
    ----------
    architecture_type : str, optional
      The type of network architecture. The three possible options are:
        - 'exp-down': creates a network that will have an exponentially
          decreasing number of nodes in each successor layer.
        - 'const': creates a network with a constant number of nodes for each
          layer
        - 'linear': creates a network with linearly decreasing node numbers for
          each successor layer
        The default is 'down'.
    n_layers : int, optional
      Number of intermediate layers. The default is 3.
    n_features : int, optional
      Number of features. The default is None.
    n_nodes : int, optional
      Number of nodes in first layer. The default is None.

    Returns
    -------
    None.

    """
    super().__init__()
    self.architecture_type = architecture_type
    self.n_layers = n_layers
    self.n_features = n_features
    self.n_nodes = n_nodes

    self._MAP = {'architecture_type': self.set_architecture_type,
                 'n_layers': self.set_n_layers,
                 'n_features': self.set_n_features,
                 'n_nodes': self.set_n_nodes}

  def __repr__(self):
    return ('Parmeter container for the generic build process\n'
            f'{self.__dict__}')

  # SETTERS:
  def set_architecture_type(self, architecture_type):
    self.architecture_type = architecture_type

  def set_n_layers(self, n_layers):
    self.n_layers = n_layers

  def set_n_features(self, n_features):
    self.n_features = n_features

  def set_n_nodes(self, n_nodes):
    self.n_nodes = n_nodes

  # HELPERS:
  def _get_archotecture_type(self, architecture_type):
    """
    Provides different aliases for all possible types and searches for
    the corresponding type (e.g. 'exp down' --> 'exp_down').

    Parameters
    ----------
    architecture_type : str
      Architecture type alias.

    Raises
    ------
    NameError
      If architecture type is not a valid alias.

    Returns
    -------
    TYPE : str
      The proper metric used for the following operations inside the class.

    """
    for TYPE in _ARCHITECTURE_TYPES:
      if architecture_type in _ARCHITECTURE_TYPES[f'{TYPE}']:
        return TYPE
    raise NameError(f"'{architecture_type}' is not a valid architecture type.")