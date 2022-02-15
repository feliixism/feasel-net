import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import rcParams as params
from matplotlib.colors import LinearSegmentedColormap

class DefaultPlot:
  def __init__(self, cmap='tab20c'):
    #colors of the University of Stuttgart
    self.colors = {'white': [1., 1., 1.],
                   'light_blue': [0/255, 190/255, 255/255],
                   'middle_blue': [0/255, 81/255, 158/255],
                   'anthrazit': [62/255, 68/255, 76/255],
                   'black': [0., 0., 0.],
                   }

    self.im_cmap = LinearSegmentedColormap.from_list('deafult',
                                                     self.get_color_list())

    self.c_cmap = plt.get_cmap(cmap)

    self.color = self.colors['middle_blue']
    self.is_tex = False

  def get_color_list(self):
    color_list = list()
    for color in self.colors:
      color_list.append(self.colors[color])
    return color_list

  def get_cycler(self, cmap=None, n=10, groups=None):
    if cmap is None:
      cmap = self.cmap
    cmap = plt.get_cmap(cmap)
    cc, ls = [], []
    ls_types = ['-', '--', '-.', ':']

    if not groups:
      groups = 1

    m = int(n/groups)

    for i in range(m):
      for ls_type in ls_types:
        for j in range(groups):
          pos = i + j * m
          cc.append(cmap(pos/n))
          ls.append(ls_type)

    default_cycler = (cycler(color=cc) +
                      cycler(linestyle=ls))

    return default_cycler

  def set_default(self):
    # default settings
    params['lines.linewidth'] = 1
    params['lines.linestyle'] = '-'
    params['image.cmap'] = self.c_cmap

    # params['axes.prop_cycle'] = self.get_cycler(cmap=self.c_cmap,
    #                                                   n=20,
    #                                                   groups=5)

    params['axes.prop_cycle'] = self.get_cycler(cmap='tab20c', n=20, groups=5)

    params['grid.linestyle'] = '--'
    params['grid.color'] = [0.5, 0.5, 0.5]
    params['grid.linewidth'] = 0.5
    params['axes.grid.which'] = 'major'
    params['axes.grid'] = True

    params['figure.dpi'] = 100

    # fontsizes
    params.update({'font.size': '10',
                   'axes.labelsize': 'x-small',
                   'axes.titlesize': 'x-small',
                   'legend.fontsize': 'x-small',
                   'xtick.labelsize': 'x-small',
                   'ytick.labelsize': 'x-small'
                   })

    # ticks
    params.update({'xtick.major.size': '2',
                   'xtick.minor.size': '1',
                   'xtick.major.pad': '2',
                   'xtick.minor.pad': '2',
                   'xtick.minor.visible': True,
                   'ytick.major.size': '2',
                   'ytick.minor.size': '1',
                   'ytick.major.pad': '2',
                   'ytick.minor.pad': '2',
                   'ytick.minor.visible': True
                   })

    # fontsizes
    params.update({'font.size': '8',
                   'axes.labelsize': 'medium',
                   'axes.titlesize': 'medium',
                   'legend.fontsize': 'small',
                   'xtick.labelsize': 'small',
                   'ytick.labelsize': 'small'
                   })

    # constrained_layout:
    params.update({'figure.constrained_layout.use': True,
                   'figure.constrained_layout.h_pad': 0.05,
                   'figure.constrained_layout.w_pad': 0.05,
                   'figure.constrained_layout.hspace': 0.05,
                   'figure.constrained_layout.wspace': 0.05,
                   })

  def get_cmap(self, **kwargs):
    """
    Enables the cmap for all plot methods. Standard is 'Greys'.

    Parameters
    ----------
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    cmap : TYPE
        DESCRIPTION.

    """
    if 'map' in kwargs:
      cmap = kwargs['map']
    else:
      cmap = self.c_cmap
    cmap = cm.get_cmap(cmap)
    return cmap

  def set_format(self, fmt):
    self.fmt = fmt

class AdvancedPlot(DefaultPlot):
  def __init__(self):
    super().__init__()

  def enable_tex(self):
    # mpl.use('pgf')
    params.update({"font.family": "serif",
                   "text.usetex": True,
                   "pgf.rcfonts": False,
                   "pgf.texsystem": 'pdflatex', # default is xetex
                   # "pgf.preamble": [r"\usepackage[T1]{fontenc}",
                   #                  r"\usepackage{mathpazo}"]
                   })
    self.is_tex = True

  def enable_pgf(self):
    # mpl.use('pgf')
    params.update({"font.family": "serif",
                   "text.usetex": True,
                   "pgf.rcfonts": False,
                   "pgf.texsystem": 'pdflatex', # default is xetex
                   # "pgf.preamble": [r"\usepackage[T1]{fontenc}",
                   #                  r"\usepackage{mathpazo}"]
                   })

  def convert_to_tex(self, obj):
    obj = np.array(obj)
    if self.is_tex:
      if obj.dtype == 'str':
        obj = np.char.replace(obj, "_", "\_")
    return obj

  def replace_char(self, obj, char1, char2):
    obj = np.array(obj)
    if obj.dtype.type == np.str_:
      obj = np.char.replace(obj, f"{char1}", f"{char2}")
    return obj