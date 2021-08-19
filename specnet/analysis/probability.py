import numpy as np

def gauss_1d(x, mu, sigma):
    gauss = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1/2 *(x - mu)**2 / sigma**2)
    return gauss
    
# def gauss_2d(x, y, mu_x = 0, mu_y = 0, sigma_x = 1, sigma_y = 1, theta = 0, 
#              a = 1, normalize = False):
#     x = x.reshape([1, len(x)])
#     y = y.reshape([len(y), 1])
#     A = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
#     B = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
#     C = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
#     gauss = a * np.exp(-(A * (x - mu_x)**2 + 2 * B * (x - mu_x) * (y - mu_y) + C * (y - mu_y)**2))
#     Sigma = np.array([[A, B], [B, C]])
#     if normalize:
#         gauss = gauss / np.sqrt((2 * np.pi)**2 * np.linalg.det(Sigma))
#     return gauss

# def gauss_2d(x, y, mu_x = 0, mu_y = 0, sigma_x = 1, sigma_y = 1, theta = 0, 
#               a = 1, normalize = False):
#     x = x.reshape([1, len(x)])
#     y = y.reshape([len(y), 1])
#     A = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
#     B = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
#     C = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
#     gauss = a * np.exp(-(A * (x - mu_x)**2 + 2 * B * (x - mu_x) * (y - mu_y) + C * (y - mu_y)**2))
#     Sigma = np.array([[A, B], [B, C]])
#     if normalize:
#         gauss = gauss / np.sqrt((2 * np.pi)**2 * np.linalg.det(Sigma))
#     return gauss

# def gauss_2d(x, y, size = (10, 10), sampling = 100):
#     data = np.array([x, y])
#     variance = np.cov(data)
#     mean = np.mean(data, axis = -1)
#     X = np.linspace(-size[0] + mean[0], size[0] + mean[0], sampling)
#     Y = np.linspace(-size[1] + mean[1], size[1] + mean[1], sampling)
#     X, Y = np.meshgrid(X, Y)
#     R = np.sqrt(X**2 + Y**2)
#     Z = 1. / (2 * np.pi * variance[0, 0] * variance[1,1]) * np.exp(-1/2 * R**2)
#     return X + mean[0], Y + mean[1], Z

def gauss_2d(x, y, size = (20, 20), sampling = (100, 100), normalize = True, a = 1):
    data = np.array([x, y], ndmin = 2)
    
    Sigma = np.cov(data)
    mu = np.mean(data, axis = -1, keepdims = True)
    x = np.linspace(-size[0], size[0], sampling[0])
    x = [x, x]
    
    if normalize:
        gauss = 1 / (2 * np.pi) * np.sqrt(np.linalg.det(Sigma)) * np.exp(-1/2 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu))
    else:
        gauss = a * np.exp(-1/2 * (sampling - mu).T @ np.linalg.inv(Sigma) @ (sampling - mu))
    
    return gauss

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# Mean vector and covariance matrix
mu = np.array([0., 1.])
Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

x_sample = np.random.normal(loc = 3, scale = 4, size = [1000])
y_sample = np.random.normal(loc = 3, scale = 2, size = [1000])

gauss = gauss_2d(x_sample, y_sample, size = (10,10))

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, cmap='twilight', shade = False, rcount = 100, ccount = 100)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')