import tensorflow as tf
import numpy as np


def add_filters(sess):
    sess.add_tensor_filter("has_nan", _has_nan)

def _has_nan(datum, tensor):
    if hasattr(tensor, 'dtype'):
        if (np.issubdtype(tensor.dtype, np.float) or np.issubdtype(tensor.dtype, np.complex) or np.issubdtype(tensor.dtype, np.integer)):
            return np.any(np.isnan(tensor))
        else:
            return False
    else:
        return False

