from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

def get_filter():
  FILTER = {"savitzky_golay": savitzky_golay,
            "gauss": gauss}
  return FILTER

def filter(X, type, *args, **kwargs):
  """
  Apply a filter to an array.

  Parameters
  ----------
  X : ndarray
    The data to be filtered.
  type : str
    The filter type to be applied.
  *args : args
    The arguments for each filter function.
  **kwargs : kwargs
    The key word arguments for each filter function.

  Returns
  -------
  y : TYPE
    DESCRIPTION.

  """
  y = get_filter()[type](*args, **kwargs)
  return y

def savitzky_golay(X, window_size=7, pol_order=3):
  """
  Apply a Savitzky-Golay filter to an array.

  Parameters
  ----------
  X : ndarray
    The data to be filtered.
  window_size : int
    The length of the filter window. Must be odd.The default is 7.
  pol_order : int
    The order of the polynomial used to fit the samples. pol_order must be less
    than window_size.

  Returns
  -------
  y : ndarray
    The filtered data.

  """
  y = savgol_filter(X, window_length=window_size, polyorder=pol_order,
                     axis=-1)
  return y

def gauss(data, sigma, mode='reflect'):
  """
  Multidimensional Gaussian filter.

  Parameters
  ----------
  data : ndarray
    The data to be filtered.
  sigma : float
    Standard deviation for Gaussian kernel.
  mode : str, optional
    The mode parameter determines how the input array is extended when the
    filter overlaps a border. By passing a sequence of modes with length equal
    to the number of dimensions of the input array, different modes can be
    specified along each axis. Default value is ‘reflect’. The valid values and
    their behavior is as follows:

    ‘reflect’ (d c b a | a b c d | d c b a)
      The input is extended by reflecting about the edge of the last pixel.
      This mode is also sometimes referred to as half-sample symmetric.

    ‘constant’ (k k k k | a b c d | k k k k)
      The input is extended by filling all values beyond the edge with the same
      constant value, defined by the cval parameter.

    ‘nearest’ (a a a a | a b c d | d d d d)
      The input is extended by replicating the last pixel.

    ‘mirror’ (d c b | a b c d | c b a)
      The input is extended by reflecting about the center of the last pixel.
      This mode is also sometimes referred to as whole-sample symmetric.

    ‘wrap’ (a b c d | a b c d | a b c d)
      The input is extended by wrapping around to the opposite edge. The
      default is 'reflect'.

  Returns
  -------
  y : ndarray
    The filtered data.

  """
  y = gaussian_filter(data, sigma=sigma, mode=mode)
  return y
