import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcursors

def click_connect(ax, features):
    mplcursors.cursor(ax).connect("add", lambda sel: sel.annotation.set_text(features[sel.target.index]))