"""
feasel.parameter.build
======================
"""

from .base import BaseParams

class BuildParamsLinear(BaseParams):
  """
  Parameter class for all build parameters.

  Attributes
  ----------
  n_components : int
    The number of prinicpal components (i.e. latent variables) :math:`q`.
  n_features : int
    The number of features :math:`n_f` that shall be extracted.
  solver : str
    The solver used for the eigenvalue problem.
  two_staged : bool
    Only for Linear Discriminant Analyses (LDAs). Decides whether a two-staged
    LDA is applied. This is needed, when the number of features is extensively
    bigger than the number of samples.
  method : str
    The method used to calculate the linear dependence (e.g. covariance) of the #
    features.
  """
  def __init__(self,
               n_components=2,
               n_features=None,
               solver='eigs',
               two_staged=False,
               method='covariance'):
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
    """
    Sets the number of components :math:`q`.

    Parameters
    ----------
    n_components : int
      The number of components.
    """
    self.n_components = n_components

  def set_n_features(self, n_features):
    """
    Sets the number of features :math:`n_f`.

    Parameters
    ----------
    n_features : int
      The number of features.
    """
    self.n_features = n_features

  def set_solver(self, solver):
    """
    Sets the solver used for the eigenvalue problem.

    Parameters
    ----------
    solver : str
      The solver used for the eigenvalue problem.

      Possible options are:

        - `eigs`: Calculates the eigenvalues solving the eigenvalue problem.
        - `svd`: Applies Singular Value Decomposition (SVD).

      .. list-table:: Solver Aliases
        :widths: 25 50
        :header-rows: 1

        * - Solver
          - Aliases
        * - `eigs`
          - `eigs`, `EIGS`, `eigenvectors`, `eigen vectors`, `eigs.`
        * - `svd`
          - `svd`, `SVD`, `singular value decomposition`
    """
    self.solver = solver

  def set_two_staged(self, two_staged):
    """
    Sets the option for a two staged solution of the eigenvalue problem. This
    is necessary for LDAs with a high features to samples ratio. The solution
    is then found, transforming the data via PCA beforehand and generating a
    lower dimensional subset (i.e. less features) and applying the LDA
    afterwards.

    Parameters
    ----------
    two_staged : bool
      Decides whether the two-staged solution for the LDA is executed or not.

    References
    ----------
      `[1] <https://epubs.siam.org/doi/10.1137/1.9781611972740.7>`_ Howland P.
      and Park H., Equivalence of Several Two-stage Methods for Linear Discriminant
      Analysis
    """
    self.two_staged = two_staged

  def set_method(self, method):
    """
    Sets the method for the calculation of the linear dependence (e.g.
    covariance) of the features.

    Parameters
    ----------
    method : str
      The method for the calculation of the linear dependence (e.g. covariance)
      of the features.

      Possible options are:

        - `covariance`: Uses the covariance matrix :math:`\Sigma` for the
          calculation of linear dependencies.
        - `correlation`: Uses the correlation matrix :math:`R` for the
          calculation of linear dependencies.

      .. list-table:: Method Aliases
        :widths: 25 50
        :header-rows: 1

        * - Method
          - Aliases
        * - `covariance`
          - `covariance`, `cov`, `cov.`, `Sigma`, `sigma`
        * - `correlation`
          - `correlation`, `corr`, `corr.`, `R`, `r`
    """
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
    _SOLVER_TYPES = {'eigs': ['eigs', 'EIGS', 'eigenvectors', 'eigen vectors',
                              'eigs.'],
                     'svd': ['svd', 'SVD', 'singular value decomposition']}
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
    _METHOD_TYPES = {'covariance': ['covariance', 'cov',
                                    'cov.', 'Sigma', 'sigma'],
                     'correlation': ['correlation', 'corr',
                                     'corr.', 'R', 'r']}
    for METHOD in _METHOD_TYPES:
      if method in _METHOD_TYPES[f'{METHOD}']:
        return METHOD
    raise NameError(f"'{method}' is not a valid normalization.")

class BuildParamsNN(BaseParams):
  """
  A parameter class for the generic construction of a `fully connected` (FC)
  neural network type.

  Attributes
  ----------
  architecture_type : str
    The type of network architecture. The three possible options are:
      - `exp-down`: creates a network that will have an exponentially
        decreasing number of nodes in each successor layer.
      - `const`: creates a network with a constant number of nodes for each
        layer
      - `linear`: creates a network with linearly decreasing node numbers for
        each successor layer
  n_layers : int
    The number of intermediate layers :math:`n_l`.
  n_features : int
    The number of features :math:`n_f`.
  n_nodes : int
    The number of nodes :math:`n_n` in the feature selection layer.
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
    return ('Parmeter container for the generic build process')

  # SETTERS:
  def set_architecture(self, architecture_type):
    """
    Sets the current architecture type to be build.

    Parameters
    ----------
    architecture_type : str
      The type of network architecture.

      Possible options are:

        - `const`: Creates a network with a constant number of nodes for each
          layer
        - `exponential`: Creates a network that will have an exponentially
          decreasing number of nodes in each successor layer.
        - `linear`: Creates a network with linearly decreasing node numbers for
          each successor layer
        - `rhombus`: Creates a network with linearly increasing and then
          decreasing node numbers for each successor layer

      .. list-table:: Architecture Type Aliases
        :widths: 25 50
        :header-rows: 1

        * - Architecture Type
          - Aliases
        * - `const`
          - `constant`, `const`
        * - `exponential`
          - `exp-down`, `exponential`, `exp.`, `exp down`,
            `exponential decrease`
        * - `linear`
          - `down`, `linear`, `linear decrease`
        * - `rhombus`
          - `up down`, `up-down`, `linear up down`, `linear-up-down`, `rhombus`

    """
    self.architecture_type = self._get_architecture(architecture_type)

  def set_n_layers(self, n_layers):
    """
    Sets the number of layers :math:`n_l` to be build.

    Parameters
    ----------
    n_layers : int
      The number of intermediate layers.
    """
    self.n_layers = n_layers

  def set_n_features(self, n_features):
    """
    Sets the number of features :math:`n_f` to be build.

    Parameters
    ----------
    n_features : int
      The number of features.
    """
    self.n_features = n_features

  def set_n_nodes(self, n_nodes):
    """
      Sets the number of nodes :math:`n_n` in the first intermediate layer.

      Parameters
      ----------
      n_nodes : int
        The number of nodes.
      """
    self.n_nodes = n_nodes

  # HELPERS:
  def _get_architecture(self, architecture_type):
    """
    Provides different aliases for all possible types and searches for
    the corresponding type (e.g. `exp down` --> `exp_down`).

    Parameters
    ----------
    architecture_type : str
      The architecture type alias.

    Raises
    ------
    NameError
      If architecture type is not a valid alias.

    Returns
    -------
    type : str
      The proper metric used for the following operations inside the class.

    """

    _ARCHITECTURE_TYPES = {'exponential': ['exp-down', 'exponential', 'exp.',
                                           'exp down', 'exponential decrease'],
                           'linear': ['down', 'linear', 'linear decrease'],
                           'rhombus': ['up-down', 'linear_up_down',
                                       'linear up down', 'linear-up-down',
                                       'up down', 'rhombus'],
                           'const': ['const', 'constant']}

    for type in _ARCHITECTURE_TYPES:
      if architecture_type in _ARCHITECTURE_TYPES[f'{type}']:
        return type
    raise NameError(f"'{architecture_type}' is not a valid architecture type.")