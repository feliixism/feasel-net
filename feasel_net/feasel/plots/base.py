from pathlib import Path
import matplotlib.pyplot as plt

from .utils import _default

class Base:
  def __init__(self, path=None):
    """
    A base Plot class that provides path information for save operations. It
    also sets default plot parameters.

    Returns
    -------
    None.

    """
    self.set_path(path)
    self.default = _default.AdvancedPlot()
    self.default.set_default()
    self.default.enable_tex()
    self.c_cmap = self.default.c_cmap
    self.im_cmap = self.default.im_cmap

  def set_path(self, path=None):
    """
    Sets the base path for where the plots are saved.

    Parameters
    ----------
    path : str
      Base plot path.

    Returns
    -------
    None.

    """
    if not path:
      self.path = Path.cwd() / 'plots'
    else:
      self.path = Path(path)

  def save(self, filename, figname=None, format='png', dpi=250):
    """
    A method that saves the current figure at the specified path.

    Parameters
    ----------
    filename : str
      The filename of the plot.
    filename : str
      The name of the figure to reactivate the specified figure. If None, the
      current figure is saved. The default is None.
    format : str, optional
      The format of the file. The default is 'pgf'.
    dpi : int, optional
      The resolution in dots per inches. The default is 100.

    Raises
    ------
    NameError
      If format is not valid.

    Returns
    -------
    None.

    """
    path = Path(filename)
    # check file-type and append if necessary:
    if path.suffix:
      format = path.suffix.split('.')[1]
    else:
      path = Path(str(path) + '.' + format)

    FMTs = ('pgf', 'png', 'pdf', 'svg', 'jpg')

    if not format in FMTs:
      raise NameError(f"Format '{format}' is invalid for save().")

    # start generating file path as a concatenation of container path and
    # filename:
    else:
      if not path.is_absolute():
        path = self.path / path

    # makedir if not already available:
    if not path.parents[0].is_dir():
      Path.mkdir(path.parents[0], parents=True, exist_ok=True)

    if figname:
      fig = plt.figure(figname) # reloads figure if figname is given

    else:
      fig = plt.gcf() # current figure

    plt.savefig(path, dpi=dpi)