from spec_net.plot import Base

a = Base()
path = a.path

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
y = x

fig = plt.figure('Test')
ax = fig.add_subplot(111)
ax.plot(x, y, label='Test1')

fig = plt.figure('Test2')
ax = fig.add_subplot(111)
ax.plot(x, y)

fig = plt.figure('Test')
ax.plot(x, 2*y, label='Test2')
ax.legend()
