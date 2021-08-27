import numpy as np
import matplotlib.pyplot as plt

def ordinary_least_squares(x, y, poly_order = 3, increasing = True):
    vander = np.vander(x, N = poly_order + 1)
    coefficients = np.matmul(np.linalg.inv(np.matmul(vander.T, vander)), np.matmul(vander.T, y))
    return coefficients

x = np.random.uniform(size = [15])
y = np.random.uniform(size = [15])
coefficients = ordinary_least_squares(x, y)
plt.scatter(x, y)
x = np.linspace(np.amin(x), np.amax(x), 100)
y = coefficients[-1] + x * coefficients[-2] + x**2 * coefficients[-3] 
plt.plot(x, y)