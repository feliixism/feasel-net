from .base import BaseParams

_SOLVER_TYPES = {'eigs': ['eigs', 'EIGS', 'eigenvectors', 'eigen vectors',
                          'eigs.'],
                 'svd': ['svd', 'SVD', 'singular value decomposition']}

_METHOD_TYPES = {'covariance': ['covariance', 'cov', 'cov.', 'Sigma', 'sigma'],
                 'correlation': ['correlation', 'corr', 'corr.', 'R', 'r']}

class Linear(BaseParams):
  def __init__(self,
               n_components=2,
               n_features=None,
               solver='eigs',
               two_staged=False,
               method='covariance'):
    """
    Parameter class for all build parameters.

    Parameters
    ----------
    n_components : int, optional
      Number of prinicpal components. The default is 2.
    n_features : int, optional
      Number of features that shall be extracted. The default is None.
    solver : str, optional
      The solver used for the eigenvalue problem. The default is 'eigs'.
    two_staged : bool, optional
      Decides whether a two-staged LDA is applied. This is needed, when the
      number of features is extensively bigger than the number of samples. The
      default is 'False'.
    method : str, optional
      The method used to calculate the linear dependence of the features. The
      default is 'covariance'.


    Returns
    -------
    None.

    """
    super().__init__()
    self.n_components = n_components
    self.n_features = n_features
    self.solver = solver
    self.two_staged = two_staged
    self.method = method

    self._MAP = {'n_components': self.set_n_components,
                 'n_features': self.set_n_features,
                 'solver': self.set_solver,
                 'two_staged': self.set_two_staged,
                 'method': self.set_method}

  def __repr__(self):
    return ('Parmeter container for the generic build process\n'
            f'{self.__dict__}')

  # SETTERS:
  def set_n_components(self, n_components):
    self.n_components = n_components

  def set_n_features(self, n_features):
    self.n_features = n_features

  def set_solver(self, solver):
    self.solver = solver

  def set_two_staged(self, two_staged):
    self.two_staged = two_staged

  def set_method(self, method):
    self.method = self._get_method(method)

  # HELPERS:
  def _get_solver_type(self, solver_type):
    """
    Provides different aliases for all possible types and searches for
    the corresponding type (e.g. 'eigenvectors' --> 'eigs').

    Parameters
    ----------
    solver_type : str
      Solver type alias.

    Raises
    ------
    NameError
      If solver type is not a valid alias.

    Returns
    -------
    TYPE : str
      The proper solver used for the following operations inside the class.

    """
    for TYPE in _SOLVER_TYPES:
      if solver_type in _SOLVER_TYPES[f'{TYPE}']:
        return TYPE
    raise NameError(f"'{solver_type}' is not a valid solver type.")

  def _get_method(self, method):
    """
    Provides different aliases for all possible methods and searches for
    the corresponding method (e.g. 'cov' --> 'covariance').

    Parameters
    ----------
    method : str
      Method alias.

    Raises
    ------
    NameError
      If method is not a valid alias.

    Returns
    -------
    METHOD : str
      The proper method used for the following operations inside the class.

    """
    for METHOD in _METHOD_TYPES:
      if method in _METHOD_TYPES[f'{METHOD}']:
        return METHOD
    raise NameError(f"'{method}' is not a valid normalization.")

class NN(BaseParams):
  """
  A parameter class for the generic construction of a 'Fully Connected' neural
  network type.

  Parameters
  ----------
  architecture_type : `str`, optional
    The type of network architecture. The three possible options are:
      - `exp-down`: creates a network that will have an exponentially
        decreasing number of nodes in each successor layer.
      - `const`: creates a network with a constant number of nodes for each
        layer
      - `linear`: creates a network with linearly decreasing node numbers for
        each successor layer
    The default is 'down'.
  n_layers : `int`, optional
    Number of `int`ermediate layers. The default is 3.
  n_features : `int`, optional
    Number of features. The default is None.
  n_nodes : `int`, optional
    Number of nodes in first layer. The default is None.

  Methods
  -------


  """
  def __init__(self,
               architecture_type='linear',
               n_layers=3,
               n_features=None,
               n_nodes=None):
    super().__init__()
    self.architecture_type = architecture_type
    self.n_layers = n_layers
    self.n_features = n_features
    self.n_nodes = n_nodes

    self._MAP = {'architecture_type': self.set_architecture,
                 'n_layers': self.set_n_layers,
                 'n_features': self.set_n_features,
                 'n_nodes': self.set_n_nodes}

  def __repr__(self):
    return ('Parmeter container for the generic build process\n'
            f'{self.__dict__}')

  # SETTERS:
  def set_architecture(self, architecture_type):
    """
    Sets the current architecture type to be build.

    Parameters
    ----------
    architecture_type : `str`
      The type of network architecture. The three possible options are:
        - 'exp-down': creates a network that will have an exponentially
          decreasing number of nodes in each successor layer.
        - 'const': creates a network with a constant number of nodes for each
          layer
        - 'linear': creates a network with linearly decreasing node numbers for
          each successor layer

    Returns
    -------
    None.

    """
    self.architecture_type = self._get_architecture(architecture_type)

  def set_n_layers(self, n_layers):
    """
    Sets the number of layers to be build.

    Parameters
    ----------
    n_layers : `int`
      Number of intermediate layers.

    Returns
    -------
    None.

    """
    self.n_layers = n_layers

  def set_n_features(self, n_features):
    """
    Sets the number of features to be build.

    Parameters
    ----------
    n_features : `int`
      Number of features.

    Returns
    -------
    None.

    """
    self.n_features = n_features

  def set_n_nodes(self, n_nodes):
    """
      Sets the number of nodes in the first intermediate layer.

      Parameters
      ----------
      n_nodes : `int`
        Number of nodes.

      Returns
      -------
      None.

      """
    self.n_nodes = n_nodes

  # HELPERS:
  def _get_architecture(self, architecture_type):
    """
    Provides different aliases for all possible types and searches for
    the corresponding type (e.g. 'exp down' --> 'exp_down').

    Parameters
    ----------
    architecture_type : `str`
      Architecture type alias.

    Raises
    ------
    NameError
      If architecture type is not a valid alias.

    Returns
    -------
    TYPE : `str`
      The proper metric used for the following operations inside the class.

    """

    _ARCHITECTURE_TYPES = {'exponential': ['exp-down', 'exponential', 'exp.',
                                           'exp down', 'exponential decrease'],
                           'linear': ['down', 'linear', 'linear decrease'],
                           'rhombus': ['up-down', 'linear_up_down',
                                       'linear up down', 'linear-up-down',
                                       'up down', 'rhombus'],
                           'const': ['const', 'constant']}

    for TYPE in _ARCHITECTURE_TYPES:
      if architecture_type in _ARCHITECTURE_TYPES[f'{TYPE}']:
        return TYPE
    raise NameError(f"'{architecture_type}' is not a valid architecture type.")