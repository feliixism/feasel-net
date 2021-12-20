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
        'figure.figsize': figsize,
        'text.usetex': True,
        'pgf.rcfonts': True,
        })
    plt.savefig(f'{path}.pgf')
    
def png(path, figsize = [12, 6.75]):
    figsize = [cm2inch(figsize[0]), cm2inch(figsize[1])]
    matplotlib.rcParams.update({
    'font.family': 'serif',
    'figure.figsize': figsize,
    })
    plt.savefig(f"{path}.png", dpi = 300)