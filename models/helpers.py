import tensorflow as tf
import collections

from math import sqrt

def reshape_using_context_window(X, context_window):
    """ X: batch_size X height X width X depth """
    WH, WW = context_window
    _, H, W, D = X.get_shape().as_list()
    N = tf.shape(X)[0]

    if H % WH != 0:
        pad = tf.zeros([N, WH - (H % WH), W, D])
        X = tf.concat(values=[X, pad], axis=1)
        H = X.get_shape().as_list()[1]

    if W % WW != 0:
        pad = tf.zeros([N, H, WW - (W % WW), D])
        X = tf.concat(values=[X, pad], axis=2)
        W = X.get_shape().as_list()[2]

    new_H = int(H/WH)
    new_W = int(W/WW)

    o1 = tf.reshape(X, [-1, H, new_W, WW*D])
    o2 = tf.transpose(o1, [0, 2, 1, 3])
    o3 = tf.reshape(o2, [-1, new_W, new_H, WH*WW*D])
    out = tf.transpose(o3, [0, 2, 1, 3])

    return out

def _single_cell(unit_type, num_units, dropout=None, name=None, residual_connection=False, residual_fn=None):
    if unit_type == "lstm":
        cell = tf.contrib.rnn.BasicLSTMCell(num_units, name=name)
    elif unit_type == "gru":
        cell = tf.contrib.rnn.GRUCell(num_units, name=name)
    elif unit_type == "layer_norm_lstm":
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, layer_norm=True)
    else:
        raise ValueError("Unknown cell type %s!" % cell_type)

    if dropout is not None:
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=1-dropout)
    
    if residual_connection:
        cell = tf.contrib.rnn.ResidualWrapper(cell, residual_fn=residual_fn)

    return cell

def _cell_list(unit_type, num_layers, num_units, dropout=None, base_name=None, num_residual_layers=0, residual_fn=None):
    cell_list = []
    for i in range(num_layers):
        single_cell = _single_cell(unit_type=unit_type,
                                   num_units=num_units,
                                   dropout=dropout,
                                   name=base_name+'_{}'.format(i+1) if (num_layers > 1 and base_name is not None) else base_name,
                                   residual_connection=(i >= num_layers - num_residual_layers),
                                   residual_fn=residual_fn)
        cell_list.append(single_cell)

    return cell_list

def create_rnn_cell(unit_type, num_layers, num_units, dropout=None, base_name=None, num_residual_layers=0, residual_fn=None):
    cell_list = _cell_list(unit_type=unit_type,
                           num_layers=num_layers,
                           num_units=num_units,
                           dropout=dropout,
                           base_name=base_name,
                           num_residual_layers=num_residual_layers,
                           residual_fn=residual_fn)

    if num_layers == 1:
        return cell_list[0]
    
    else:
        return tf.contrib.rnn.MultiRNNCell(cell_list)

def create_attention_images_summary(attention_matrix, name, max_outputs=32):
    """create attention image and attention summary."""
    # Reshape to (batch, width, target_size, color)
    attention_images = tf.expand_dims(tf.transpose(attention_matrix, [1, 0, 2]), -1)
    tf.summary.image(name, attention_images, max_outputs=max_outputs)

def get_label(i):
    if i % 2 == 0:
        return "weights"
    else:
        return "biases"

def get_trainable_weights(cell):
    """
    Returns trainable weights for an rnn cell potentially wrapped within a DropoutWrapper
    """
    if type(cell) == tf.nn.rnn_cell.DropoutWrapper:
        return cell.wrapped_cell.trainable_weights

    return cell.trainable_weights

def get_trainable_variables(cell):
    """
    Returns names of trainable variables for an rnn cell potentially wrapped within a DropoutWrapper
    """
    if type(cell) == tf.nn.rnn_cell.DropoutWrapper:
        cell = wrapped_cell

    return [t.name.split(':')[0] for t in cell.trainable_variables]

def encoder_params_helper(num_layers, unit_type, direction, num_units, num_residual_layers=0, verbose=False):
    EncoderParams = collections.namedtuple('EncoderParams', ["encoder_num_layers", "encoder_unit_type", "encoder_type",
                                                             "encoder_num_units", "encoder_num_residual_layers",
                                                             "verbose"])

    params = EncoderParams(num_layers, unit_type, direction, num_units, num_residual_layers, verbose)

    return params

def put_kernel_on_grid (kernel, pad = 1):
    """
    Taken from: https://gist.github.com/kukuruza/03731dc494603ceab0c5
    
    Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    
    Args:
        kernel: tensor of shape [Y, X, NumChannels, NumKernels]
        pad: number of black pixels around each filter (between them)
    Return:
        Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    """

    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: 
                    print('Who would enter a prime number of filters')
                return (i, int(n / i))
    
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))
    
    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard

    return x

def variable_summaries(var, scope):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(scope):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
