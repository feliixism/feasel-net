from .base import BaseParams

_SOLVER_TYPES = {'eigs': ['eigs', 'EIGS', 'eigenvectors', 'eigen vectors',
                          'eigs.'],
                 'svd': ['svd', 'SVD', 'singular value decomposition']}

_METHOD_TYPES = {'covariance': ['covariance', 'cov', 'cov.', 'Sigma', 'sigma'],
                 'correlation': ['correlation', 'corr', 'corr.', 'R', 'r']}

class BuildParams(BaseParams):
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