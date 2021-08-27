import matplotlib
import matplotlib.pyplot as plt

def cm2inch(val):
    return val/2.54

def pgf(path, figsize = [12, 6.75]):
    figsize = [cm2inch(figsize[0]), cm2inch(figsize[1])] 
    # latex
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 12,
        'figure.figsize': figsize,
        'text.usetex': True,
        'legend.fontsize': 10,
        'pgf.rcfonts': True,
        })
    plt.savefig(f'{path}.pgf')
    
def png(path, figsize = [12, 6.75]):
    figsize = [cm2inch(figsize[0]), cm2inch(figsize[1])]
    matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 8,
    'axes.linewidth': 0.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.figsize': figsize,
    'legend.fontsize': 8
    })
    plt.savefig(f"{path}.png", dpi = 300)