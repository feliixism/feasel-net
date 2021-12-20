import numpy as np
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import cm, rc
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
        self.c_cmap = cmap
        
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
        mpl.rcParams['lines.linewidth'] = 1
        mpl.rcParams['lines.linestyle'] = '-'
        mpl.rcParams['image.cmap'] = self.c_cmap
        
        mpl.rcParams['axes.prop_cycle'] = self.get_cycler(cmap=self.c_cmap, 
                                                          n=20, 
                                                          groups=5)
        
        mpl.rcParams['grid.linestyle'] = '--'
        mpl.rcParams['grid.color'] = [0.5, 0.5, 0.5]
        mpl.rcParams['grid.linewidth'] = 0.5
        mpl.rcParams['axes.grid'] = True
        
        mpl.rcParams['figure.dpi'] = 100
        
        # fontsizes
        mpl.rcParams.update({'font.size': '10',
                             'axes.labelsize': 'x-small',
                             'axes.titlesize': 'x-small',
                             'legend.fontsize': 'x-small',
                             'xtick.labelsize': 'x-small',
                             'ytick.labelsize': 'x-small'
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
       rc('font', **{'family': 'DejaVu Sans', 
                      'serif': ['Computer Modern']})
       rc('text', usetex=True)
       self.is_tex = True
      
    def enable_pgf(self):
        mpl.use('pgf')

        # update latex preamble
        mpl.rcParams.update({"font.family": "serif",
                             "text.usetex": True,
                             "pgf.rcfonts": False,
                             "pgf.texsystem": 'pdflatex', # default is xetex
                             "pgf.preamble": [r"\usepackage[T1]{fontenc}",
                                              r"\usepackage{mathpazo}"
                                              ]
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
 
