import numpy as np
import copy

def append_start_token(y, token_id):
    return [np.append(token_id, yi) for yi in y]

def append_end_token(y, token_id, except_max=False):
    if except_max:
        max_len = max([len(yi) for yi in y])
        new_y = []
        for yi in y:
            if len(yi) != max_len:
                new_y.append(np.append(yi, token_id))
            else:
                new_y.append(yi)
        y = new_y
    else:
        y = [np.append(yi, token_id) for yi in y]

    return y

# pad a (nested) list to some maximum length
def pad_labels(y, max_len, token_id):
    y = [post_pad_sequence(yi, max_len, value=token_id) for yi in y]

    return np.array(y)

# pre-pad a sequence to a maximum length
def pre_pad_sequence(seq, max_len, value=0):
    add_zeros = max_len - len(seq)
    new_seq = [value for i in range(add_zeros)]
    new_seq.extend(seq)
    
    return new_seq

# post-pad a sequence to a maximum length
def post_pad_sequence(seq, max_len, value=0):
    seq_copy = list(copy.deepcopy(seq))
    add_zeros = max_len - len(seq_copy)
    new_seq = [value for i in range(add_zeros)]
    seq_copy.extend(new_seq)

    return seq_copy

# create a one-hot vector
def one_hot(seq, num_classes):
    vec = np.zeros([seq.shape[0], num_classes])
    vec[np.arange(0,seq.shape[0]),seq] = 1
    
    return vec

# create a sparse tensor from a 2d array
def to_sparse(array):
    indices = []
    values  = []
    shape   = [len(array), 0]       # shape[1] must be the maximum length of array

    for (col_num, row) in enumerate(array): 
        if len(row) > shape[1]:
            shape[1] = len(row)     # update shape[1] to the maximum length

        for (row_num, value) in enumerate(row):
            indices.append([col_num, row_num])
            values.append(value)

    return indices, values, shape

# convert a sparse tensor into a 2d array
def from_sparse(sparse, batch_size):
    output_array = [[] for i in range(batch_size)]

    for (i, index) in enumerate(sparse.indices):
        output_array[index[0]].append(sparse.values[i])    # index[0] is the batch_number

    return output_array
