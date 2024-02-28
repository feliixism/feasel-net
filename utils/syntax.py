"""
feasel.utils.syntax
===================
"""

def update_kwargs(object, class_name = None, **kwargs):
    add_kwargs(object)
    check_kwargs(object, class_name = class_name, **kwargs)

def check_kwargs(object, class_name = None, **kwargs):
    # Update __dict__ but only for keys that have been predefined 
    # (silently ignore others)
    allowed_keys = object._kwargs
    
    object.__dict__.update((key, value) for key, value in kwargs.items() 
                           if key in list(allowed_keys))
    
    # To NOT silently ignore rejected keys
    if class_name == object.__class__.__name__:
        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor: {}".format(rejected_keys))

def add_kwargs(object):
    
    #only allow external variables (internal are indicated by a leading underscore)
    allowed_keys = list(object.__dict__.keys())
    
    allowed_keys = del_internal_kwargs(allowed_keys)
    
    if hasattr(object, "_kwargs"):
        object._kwargs = object._kwargs | set(allowed_keys)
    else:
        object._kwargs = set(allowed_keys)
        
def del_internal_kwargs(kwargs):
    while True:
        c = 0
        for key in kwargs:
            if key[0] == "_":
                kwargs.remove(key)
                c += 1
        if c == 0:
            break
    return kwargs

def add_params(d, c):
    """
    Adds class variables to an existing dictionary.

    Parameters
    ----------
    d : dict
        The dictionary of the original parameters.
    c : class
        The class with its class variables that is added to the dictionary.

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    """
    keys, values = c.__dict__.keys(), list(c.__dict__.values())
    for i, key in enumerate(keys):
        d[f"{key}"] = values[i]
    return d