from params import params as P
import numpy as np
import os

def float32(k):
    return np.cast['float32'](k)

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
        from http://goo.gl/DZNhk
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def make_dir_if_not_present(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
