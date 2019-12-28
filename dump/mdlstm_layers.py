import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import LSTMStateTuple
import tensorflow.contrib.slim as slim

from models.model_class import variable_summaries

def layer_normalization(tensor, scope=None, eps=1e-5):
    """ Adapted from https://github.com/areiner222/MDLSTM/blob/master/md_lstm.py """
    """ Layer normalizes a 2D tensor along its second axis """

    assert len(tensor.get_shape()) == 2

    mean, var = tf.nn.moments(tensor, [1], keep_dims=True)

    if not isinstance(scope, str):
        scope = ''

    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale', shape=[tensor.get_shape()[1]], initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift', shape=[tensor.get_shape()[1]], initializer=tf.constant_initializer(0))

    normalized_tensor = (tensor - mean)/tf.sqrt(var + eps)

    return normalized_tensor * scale + shift

def get_features_using_context_window(X, context_window):
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

class MDLSTM():
    def __init__(self, hidden_dim, context_window_size, scope=None, use_peepholes=True):
        self.hidden_dim = hidden_dim
        self.context_window_size = context_window_size
        self.use_peepholes = use_peepholes
        if not isinstance(scope, str):
            scope = ''
        self.scope = scope
          
    def step_forward(self, features, prev_hidden_states, prev_cell_states):
        H = self.hidden_dim
        h1, h2 = prev_hidden_states
        c1, c2 = prev_cell_states
        
        prod = slim.fully_connected(tf.concat([features, h1, h2], axis=1), 5*H, activation_fn=None)
        
        i, f1, f2, o, g = tf.split(value=prod, num_or_size_splits=5, axis=1)
        
        if self.use_peepholes:
            i += slim.fully_connected(tf.concat([c1, c2], axis=1), H, activation_fn=None, biases_initializer=None)
            f1 += slim.fully_connected(c1, H, activation_fn=None, biases_initializer=None)
            f2 += slim.fully_connected(c2, H, activation_fn=None, biases_initializer=None)
                   
        i = tf.nn.sigmoid(layer_normalization(i, scope='i/'))
        f1 = tf.nn.sigmoid(layer_normalization(f1, scope='f1/'))
        f2 = tf.nn.sigmoid(layer_normalization(f2, scope='f2/'))
        g = tf.nn.tanh(layer_normalization(g, scope='g/'))

        next_cell_state = f1*c1 + f2*c2 + i*g
        
        if self.use_peepholes:
            o += slim.fully_connected(next_cell_state, H, activation_fn=None, biases_initializer=None) 

        o = tf.nn.sigmoid(layer_normalization(o, scope='o/'))

        next_hidden_state = o*tf.nn.tanh(layer_normalization(next_cell_state, scope='cell/'))

        return next_hidden_state, LSTMStateTuple(next_hidden_state, next_cell_state)

    def forward(self, X):
        """ Inspired in part by https://github.com/areiner222/MDLSTM/blob/master/md_lstm.py """
        """ X: batch_size X height X width X channels """

        """ create H*W arrays """
        with tf.variable_scope(self.scope):
            _, H, W, C = X.get_shape().as_list()
            N = tf.shape(X)[0]

            X = tf.reshape(tf.transpose(X, [1,2,0,3]), [-1, C])
            X = tf.split(X, H*W, axis=0)

            """ create dynamic-sized arrays with timesteps = H*W """
            inputs = tf.TensorArray(dtype=tf.float32, size=H*W).unstack(X)
            states = tf.TensorArray(dtype=tf.float32, size=H*W+1, clear_after_read=False)
            outputs = tf.TensorArray(dtype=tf.float32, size=H*W)

            """ initialiaze states to zero  """
            states = states.write(H*W, LSTMStateTuple(tf.zeros([N, self.hidden_dim], tf.float32),
                                                      tf.zeros([N, self.hidden_dim], tf.float32)))

            """ define counter """
            t = tf.constant(0)

            """ define operations at each time step """
            def body(t_, outputs_, states_):"""TODO: check if first state should use tf.less instead of tf.less_equal"""
                states_1 = tf.cond(tf.less_equal(t_, tf.constant(W)),
                                   lambda: states_.read(H*W),
                                   lambda: states_.read(t_ - tf.constant(W)))
                states_2 = tf.cond(tf.equal(t_ % W, tf.constant(0)),
                                   lambda: states_.read(H*W),
                                   lambda: states_.read(t_ - tf.constant(1)))
     
                prev_hidden_states = LSTMStateTuple(states_1[0], states_2[0])
                prev_cell_states = LSTMStateTuple(states_1[1], states_2[1])

                out, state = self.step_forward(inputs.read(t_), prev_hidden_states, prev_cell_states)
                outputs_ = outputs_.write(t_, out)
                states_ = states_.write(t_, state)

                return t_+1, outputs_, states_

            """ define condition for while loop """
            def condition(t_, outputs_, states_):
                return tf.less(t_, tf.constant(H*W))

            """ run while loop """
            _, outputs, states = tf.while_loop(condition, body, [t, outputs, states], parallel_iterations=1)

            """ stack outputs and states to get tensor and reshape outputs appropriately """
            outputs = outputs.stack()
            states = states.stack()

            outputs = tf.transpose(tf.reshape(outputs, [H, W, -1, self.hidden_dim]), [2,0,1,3])

        return outputs, states

class MDLSTM_Wrapper_Class():
    def __init__(self, hidden_dim, context_window_size, scope=None, use_peepholes=True):
        self.hidden_dim = hidden_dim
        self.context_window_size = context_window_size
        self.scope = scope
        
        self.top_left = MDLSTM(hidden_dim, context_window_size, scope + '/top_left', use_peepholes)
        self.bottom_left = MDLSTM(hidden_dim, context_window_size, scope + '/bottom_left', use_peepholes)
        self.top_right = MDLSTM(hidden_dim, context_window_size, scope + '/top_right', use_peepholes)
        self.bottom_right = MDLSTM(hidden_dim, context_window_size, scope + '/bottom_right', use_peepholes)
        
    def __call__(self, X):
        """ X: batch_size X height X width X channels """

        if len(X.get_shape().as_list()) == 3:
            X = tf.expand_dims(X, axis=3)
        
        """ Scan in all four directions by flipping image """
        """ features: batch_size X height X width X length of feature vector """
        features_top_left = get_features_using_context_window(X, self.context_window_size)
        features_bottom_left = tf.image.flip_up_down(features_top_left)
        features_top_right = tf.image.flip_left_right(features_top_left)
        features_bottom_right = tf.image.flip_left_right(features_bottom_left)

        """ Get output """
        out_top_left, _ = self.top_left.forward(features_top_left)
        out_bottom_left, _ = self.bottom_left.forward(features_bottom_left)
        out_top_right, _ = self.top_right.forward(features_top_right)
        out_bottom_right, _ = self.bottom_right.forward(features_bottom_right)
        
        variable_summaries(out_top_left, self.scope+'/out_top_left')
        variable_summaries(out_bottom_left, self.scope+'/out_bottom_left')
        variable_summaries(out_top_right, self.scope+'/out_top_right')
        variable_summaries(out_bottom_right, self.scope+'/out_bottom_right')
        
        out = tf.concat([out_top_left, out_bottom_left, out_bottom_left, out_bottom_right], 3)

        return out

class FullyConnectedWithContextWindow():
    """ Implements a fully connected layer with context window """
    def __init__(self, hidden_dim, context_window_size, activation, scope=None):
        self.hidden_dim = hidden_dim
        self.context_window_size = context_window_size
        self.activation = activation
        if not isinstance(scope, str):
            scope = ''
        self.scope = scope
           
    def __call__(self, X):
        """ X: batch_size X height X width X depth """
        features = get_features_using_context_window(X, self.context_window_size) 
        out = slim.fully_connected(features, self.hidden_dim, activation_fn=None, scope=self.scope)
        
        return self.activation(out)
