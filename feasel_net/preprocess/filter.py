import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.signal import savgol_filter

# filter_dictionary = {"savitzky_golay": lambda: savitzky_golay(*args, **kwargs),
#                      "gauss": lambda: gauss(*args, **kwargs),
#                      "median": lambda: median(*args, **kwargs),
#                      "uniform": lambda: uniform(*args, **kwargs),
#                      "shift_n_add": lambda: shift_n_add(*args, **kwargs)}

class Filter:
    def __init__(self):
        return
    
    def filter(self, data, type, *args, **kwargs):
        if hasattr(self, type) and callable(getattr(self, type)):
            func = getattr(self, f"{type}")
            return func(data, *args, **kwargs)

    def savitzky_golay(self, data, size, n_polynom):
        filtered = savgol_filter(data, size, n_polynom, axis = -1)
        return filtered
    
    def gauss(self, data, sigma):
        filtered = gaussian_filter(data, sigma)
        return filtered
    
    def median(self, data, size):
        filtered = median_filter(data, size)
        return filtered        
        
    def uniform(self, data, size):
        filtered = uniform_filter(data, size)
        return filtered
    
    def shift_n_add(self, data, size):
        smooth_plot = np.zeros(len(data))
        for i in range(-int(np.floor(size / 2)), int(np.floor(size / 2) + 1)):
            if i == 0:
                pass
            else:
                if i > 0:
                    shifted = np.append((np.ones(len(data[-i:])) * data[i]), data[:-i])
                elif i < 0:
                    shifted = np.append(data[:i], (np.ones(len(data[i:])) * data[i]))
                smooth_plot += (data + shifted) / 2
        filtered = smooth_plot / size
        return filtered
            