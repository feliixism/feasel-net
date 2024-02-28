"""
feasel.plot.utils._visualization
================================
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import eigh
import numpy as np

def update_label(old_label, exponent_text):
    if exponent_text == "":
        return old_label

    try:
        units = old_label[old_label.index("[") + 1:old_label.rindex("]")]
    except ValueError:
        units = ""
    label = old_label.replace("[{}]".format(units), "")

    exponent_text = exponent_text.replace("\\times", "")

    return "{} [{} {}]".format(label, exponent_text, units)

def format_label_string_with_exponent(ax, axis='both'):
    """ Format the label string with the exponent from the ScalarFormatter """
    ax.ticklabel_format(style="sci")

    axes_instances = []
    if axis in ['x', 'both']:
        axes_instances.append(ax.xaxis)
    if axis in ['y', 'both']:
        axes_instances.append(ax.yaxis)

    for ax in axes_instances:
        ax.major.formatter._useMathText = True
        plt.draw() # Update the text
        exponent_text = ax.get_offset_text().get_text()
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        ax.set_label_text(update_label(label, exponent_text))

def contour_gauss(ax, sigma_x, sigma_y, mu_x = 0, mu_y = 0, angle = 0, color = "k"):
    ellipse = mpl.patches.Ellipse((mu_x, mu_y),
                                  sigma_x,
                                  sigma_y,
                                  angle = angle,
                                  color = color,
                                  linestyle = "-.",
                                  fill = False,
                                  zorder = 0)
    ax.add_patch(ellipse)

def confidence(ax, covariance_matrix, mu = [0, 0], s=3, color = "k"):
    evals, evecs = eigh(covariance_matrix)
    evals = np.flip(evals)
    evecs = np.flip(evecs.T, 0)
    angle = np.arctan(evecs[0, 0] / evecs[0, 1]) * 180 / np.pi
    try:
        for i in range(len(s)):
            contour_gauss(ax, sigma_x = 2 * np.sqrt(evals[0]) * s[i],
                          sigma_y = 2 * np.sqrt(evals[1]) * s[i],
                          mu_x = mu[0], mu_y = mu[1], angle = 90 - angle,
                          color = color)
    except:
        contour_gauss(ax, sigma_x = 2 * np.sqrt(evals[0]) * s,
                      sigma_y = 2 * np.sqrt(evals[1]) * s, mu_x = mu[0],
                      mu_y = mu[1], angle = 90 - angle, color = color)


