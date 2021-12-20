import numpy as np

def get_indices(array):
    """
    Provides a list with the indices of an array that indicates where to find 
    the data for each particular class or label.

    Parameters
    ----------
    array : ndarray
        The array with label entries.

    Returns
    -------
    l_indices : list
        A list with the indices.

    """
    labels = np.unique(array, axis=0)

    l_indices = []
    for i, label in enumerate(labels):
        l_indices.append(np.argwhere(array == label))
    
    return l_indices

def categorical(array):
    """
    Converts an one-hot encoded array into a categorical array. If the array is
    categorical already, it will be manipulated such that the neural network 
    can interpret the labels, i.e. it will rename the classes into an 
    enumeration of integers starting from 0 to the number of classes n_c - 1 
    (is obsolete when already happened).

    Parameters
    ----------
    array : ndarray
        The array with label entries.

    Returns
    -------
    categorical : ndarray
        An array of categorical labels.
    """
    # it will convert the label array in an interpretable label array for the
    # neural network
    l_indices = get_indices(array) # provides a list of indices
    
    # creates the empty categorical array:
    categorical = np.zeros(len(array)) 
    for i, indices in enumerate(l_indices):
        # array will be the i-th class in the specific indices
        categorical[indices] = i
    
    return categorical

def one_hot(array):
    """
    Converts a categorical array into an one-hot encoded array. If the array is
    one-hot-encoded already, it will be returned unmanipulated.

    Parameters
    ----------
    array : ndarray
        The array with label entries.

    Returns
    -------
    one_hot : ndarray
        An array of the one-hot encoded labels.
    """
    l_indices = get_indices(array) # provides a list of indices
        
    # creates the empty one-hot encoded array:
    one_hot = np.zeros([len(array), len(l_indices)]) 
    for i, indices in enumerate(l_indices):
        # array will be one, only if it is found in the i-th class
        one_hot[indices, i] = 1 
    
    return one_hot

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

def min_max(array, 
                  axis=None, 
                  a_max=1., 
                  a_min=0.):
    """
    Scales the original array such that it fits in between a maximum value 
    a_max and a minimum value a_min. 

    Parameters
    ----------
    array : ndarray
        The original data array.
    axis : int, optional
        The axis along which the scaling is undertaken. If 'None', it will 
        scale the complete array. The default is None.
    a_max : float, optional
        The new maximum value of the data after scaling. The default is 1.
    a_min : TYPE, optional
        The new minimum value of the data after scaling. The default is 0.

    Returns
    -------
    scaled : ndarray
        The scaled array.

    """
    # creates an empty container in the shape of the original array
    scaled = np.zeros(array.shape)
    
    # if there is an axis specified, it will take the values along this axis
    # and apply the scaling along the resulting sub-array 
    if axis is not None:
        n = array.shape[axis] # number of sub arrays
        
        for i in range(n):
            # min-max scaling of the sub-arrays
            sub_array = np.take(array, i, axis=axis)
            _a_max = np.amax(sub_array)
            _a_min = np.amin(sub_array)
            scale = (a_max-a_min) / (_a_max-_a_min) # the scaling factor
            offset = a_max - scale*_a_max # an offset after scaling
            y = scale*sub_array + offset
            scaled = np.insert(scaled, i, y, axis=axis)
        scaled = np.take(scaled, np.arange(n), axis=axis)
    
    else:
        _a_max = np.amax(array)
        _a_min = np.amin(array)
        scale = (a_max - a_min) / (_a_max - _a_min) # the scaling factor
        offset = a_max - scale * _a_max # an offset after scaling
        scaled = scale * array + offset
    
    return scaled

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
