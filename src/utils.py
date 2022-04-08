import numpy as np
import os
import random
import torch
import inspect
from collections import Counter

def seed_everything(seed=42):
    '''
    
    Function to put a seed to every step and make code reproducible
    Input:
    - seed: random state for the events 
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_attributes_config(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            pr[name] = value
    return pr

def load_object_from_dict(dict_path):
    from collections import namedtuple
    import pickle
    file = open(dict_path, "rb")
    dictionary  = pickle.load(file)
    return namedtuple("config", dictionary.keys())(*dictionary.values())