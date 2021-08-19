import numpy as np

def get_labels(label_array):
    labels = np.unique(label_array)
    return labels

def get_n_labels(label_array):
    labels = get_labels(label_array)
    n_labels = len(labels)
    return n_labels

def sparse_labels(label_array):
    labels = get_labels(label_array)
    sparse_labels = np.empty(label_array.shape)
    for label1 in range(len(labels)):
        for label2 in range(len(label_array)):
            if labels[label1] == label_array[label2]:
                sparse_labels[label2] = label1
    sparse_labels = sparse_labels.astype(int)
    return sparse_labels

def one_hot_labels(label_array):
    labels = get_labels(label_array)
    one_hot_labels = np.zeros([len(label_array), len(labels)])
    for label1 in range(len(labels)):
        for label2 in range(len(label_array)):
            if labels[label1] == label_array[label2]:
                one_hot_labels[label2, label1] = 1
    one_hot_labels = one_hot_labels.astype(int)
    return one_hot_labels

def shape_ann(x, y):
    if len(x.shape) != 2:
        old_shape = x.shape
        x = x.reshape([len(y), int(len(x.flatten()) / len(y))])
        print(f"Shape {old_shape} is converted to shape {x.shape}.")
    else:
        print(f"Shape {x.shape} is already correct.")
    return x

def shape_1d_cnn(x, y, features = 1):
    if len(x.shape) != 3:
        old_shape = x.shape
        x = x.reshape([len(y), int(len(x.flatten()) / (len(y) * features)), features])
        print(f"Shape {old_shape} is converted to shape {x.shape}.")
    else:
        print(f"Shape {x.shape} is already correct.")
    return x

def min_max_scale(arr, axis = None, a_max = 1, a_min = 0):
    new_arr = np.zeros(arr.shape)
    if axis is not None:
        n = arr.shape[axis]
        for i in range(n):
            x = np.take(arr, i, axis = axis)
            _a_max = np.amax(x)
            _a_min = np.amin(x)
            scale = (a_max - a_min) / (_a_max - _a_min)
            offset = a_max - scale * _a_max
            y = scale * x + offset
            new_arr = np.insert(new_arr, i, y, axis = axis)
        new_arr = np.take(new_arr, np.arange(n), axis = axis)
    else:
        _a_max = np.amax(arr)
        _a_min = np.amin(arr)
        scale = (a_max - a_min) / (_a_max - _a_min)
        offset = a_max - scale * _a_max
        new_arr = scale * arr + offset
    return new_arr

def standardize(arr, axis = -1):
    """
    Calculates the standardized score of the sample features.

    z = (x - x_bar) / s

    where x_bar is the arithmetic mean and s the standard deviation for each 
    feature. The features are described by the given axis.
    
    Parameters
    ----------
    arr : np-array (float)
        The original data array.
    axis : TYPE, optional
        Determines the axis of the features. The default is -1.

    Returns
    -------
    new_arr : np-array (float)
        The standard score of the original data array.

    """
    new_arr = np.zeros(arr.shape)
    
    n = arr.shape[axis]
    for i in range(n):
        x = np.take(arr, i, axis = axis)
        mu = np.mean(x)
        sigma = np.sqrt(np.var(x))
        y = (x - mu) / sigma
        new_arr = np.insert(new_arr, i, y, axis = axis)
    new_arr = np.take(new_arr, np.arange(n), axis = axis)
    return new_arr
