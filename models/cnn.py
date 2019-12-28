import tensorflow as tf

from models.helpers import put_kernel_on_grid, variable_summaries

__all__ = ["CNN"]

class CNN():
    """
    This class implements a convolutional neural network of arbitrary layers.
    """
    def __init__(self, params):
        """
        Args:
            params: The paramters of the model.
        """
        self.params = params

    def __call__(self, X, X_seq_len, is_training):
        """
        Args:
            X: the inputs shaped [batch_size, width, height, colors]
            X_seq_len: the true width of each input shaped [batch_size]
            is_training: whether the model is training or inferring, a boolean
        """
        params = self.params

        sizes_f = params.cnn_filter_sizes
        num_f = [X.get_shape().as_list()[-1]]
        num_f.extend(params.cnn_num_filters)

        activation_fn = _get_activation_fn(params.cnn_activation)

        out = X
        for i in range(params.cnn_num_layers):
            # For residual connections assume that the input and output of the convolutional layer
            # have the same dimensions except for the color channel (i.e padding is "SAME")
            #if params.cnn_residual_layers[i]:
            if i >= params.cnn_num_layers - params.cnn_num_residual_layers:
                if num_f[i] != num_f[i+1] or params.pool_sizes != (1,1) or params.pool_strides != (1,1):
                    # Need to project each value if the input and output color channels will be different
                    # Also, need to use pool strides so as to ensure that residual connection and the conv
                    # layer output have the same dimensions
                    res_kernel = _get_kernel(1, 1, num_f[i], num_f[i+1], "residual_kernel_{}".format(i))
                    residual_connection = tf.nn.conv2d(out,
                                                       res_kernel,
                                                       (1, params.pool_strides[i][0], params.pool_strides[i][1], 1),
                                                       params.pool_paddings[i])
                else:
                    residual_connection = out

            kernel = _get_kernel(sizes_f[i], sizes_f[i], num_f[i], num_f[i+1], "kernel_{}".format(i))
            out = tf.nn.conv2d(out,
                               kernel,
                               (1, params.cnn_strides[i][0], params.cnn_strides[i][1], 1),
                               params.cnn_paddings[i])
            if params.pool_sizes != (1,1) or params.pool_strides != (1,1):
                out = tf.nn.max_pool(out,
                                     (1, params.pool_sizes[i][0], params.pool_sizes[i][1], 1),
                                     (1, params.pool_strides[i][0], params.pool_strides[i][1], 1),
                                     params.pool_paddings[i])
            out = activation_fn(out)

            if params.do_batch_norm[i]:
                # Batch normalization should always be used after the activation function, especially
                # when they are ReLUs. This is because batch normalization mean centers the data and 
                # so the subsequent RelU will kill half the neurons ultimately leading to the vanishing
                # gradients problem. There is however one exception which is in the case of ResNets where
                # we have an alternative path, so all neurons do not necessarily get killed off. In that
                # case, there seems to be some debate regarding whether to apply batch normalization before
                # or after the activation (see for e.g: https://www.reddit.com/r/MachineLearning/comments
                # /67gonq/d_batch_normalization_before_or_after_relu/)
                
                # Use update_collections=None because otherwise there seems to be a problem when using
                # batch normalization and lstm/gru layers together. 
                # See: https://github.com/tensorflow/tensorflow/issues/14357
                out = tf.contrib.layers.batch_norm(out,
                                                   is_training=is_training,
                                                   trainable=True,
                                                   updates_collections=None)

            if i >= params.cnn_num_layers - params.cnn_num_residual_layers:
                out = tf.add(out, residual_connection)

            if params.use_dynamic_lengths:
                # Decrease the sequence lengths depending on the strides that were taken.
                # Note that this will only work correctly for non-overlapping convolutional
                # and pooling layers, i.e. when the filter/pool sizes and strides are equal
                # May not also work when residual connections are present.
                X_seq_len = tf.ceil(X_seq_len/(params.pool_strides[i][0]*params.cnn_strides[i][0]))

            if params.verbose:
                variable_summaries(kernel, "kernel_{}".format(i))

                # Only visualize first layer kernel
                if i == 0:
                    kernel_grid = put_kernel_on_grid (kernel)
                    tf.summary.image("kernel_grid_{}".format(i), kernel_grid, max_outputs=1)
 
                if i >= params.cnn_num_layers - params.cnn_num_residual_layers:
                    variable_summaries(res_kernel, "res_kernel_{}".format(i))

                variable_summaries(out, "out_{}".format(i))

        if params.use_dynamic_lengths:
            max_time = tf.dtypes.cast(tf.shape(out, out_type=tf.int64)[1], tf.float64)
            X_seq_len = tf.dtypes.cast(tf.clip_by_value(X_seq_len, 0, max_time), tf.int32)

        return out, X_seq_len 

def _get_kernel(h, w, c, f, name):
    kernel = tf.get_variable(name, shape=[h,w,c,f], trainable=True)

    return kernel

def _get_activation_fn(activation):
    if activation == "leaky_relu":
        activation_fn = tf.nn.leaky_relu
    elif activation == "relu":
        activation_fn = tf.nn.relu
    else:
        raise ValueError("Unknown activation %s" % activation)

    return activation_fn
