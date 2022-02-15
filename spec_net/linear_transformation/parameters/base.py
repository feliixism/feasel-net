class BaseParams:
  def __init__(self):
    """
    This is the base parameter class that provides an update function for all
    kwargs used to instantiate each parameter class ('build', 'callback',
    'data' and 'train').

    Returns
    -------
    None.

    """
    self._MAP = {}

  def update(self, key):
    return self._MAP[key]