"""
feasel.parameter.base
=====================
"""

class BaseParams:
  def __init__(self):
    """
    This is the base parameter class that provides an update function for all
    ``**kwargs`` used to instantiate each parameter class given in
    :mod:`feasel.parameter.build`, :mod:`feasel.parameter.callback`,
    :mod:`feasel.parameter.data`, and :mod:`feasel.parameter.train`.
    """
    self._MAP = {}

  def update(self, kwarg):
    """
    Updates the arguments of any arbitrary parameter specified by one of the
    aforementioned classes.

    Parameters
    ----------
    kwarg : kwarg
      A single keyword argument that matches with the attributes of one of the
      sub-container classes.
    """
    return self._MAP[kwarg]