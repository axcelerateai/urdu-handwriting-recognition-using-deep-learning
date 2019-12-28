import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.helpers import reshape_using_context_window, variable_summaries
from models.model_class import Model
from models.md_dynamic_rnn import multidimensional_dynamic_rnn
from models.mdlstm_cell import MDLSTMCell

class Config_MDLSTM_CTC:
    """
    Holds model hyperparameters and data information for the MDLSTM_CTC model.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    char_or_lig = 'char'
    use_segmented_images = True
    use_dynamic_lengths = True
    append_start_token = False

    if use_segmented_images:
        data_folder = 'data/segmented_cc'
    else:
        data_folder = 'data/UPTI'

    restore_previous_model = True
    restore_best_model = False
    name = "MDLSTM_CTC"
    if use_segmented_images:
        save_dir = 'trained_models/'+name+'/Segmented/'
    else:
        save_dir = 'trained_models/'+name+'/Unsegmented/'
    verbose = True

    num_train = None
    batch_size = 32
    num_epochs = 100
    print_every = 10

    if use_segmented_images:
        image_size = (128, 64)
        image_width_range = None
    else:
        #image_size = (200,64)  
        image_size = (800,32)
        #image_size = (2400,128)
        image_width_range = 50
    
    optimizer = tf.train.RMSPropOptimizer 
    lr = 0.001*10

    use_dropout = False
    clip_gradients = False
    if clip_gradients:
        max_grad_norm = 5 
    anneal_lr = False
    if anneal_lr:
        anneal_lr_every = 20000
        anneal_lr_rate = 0.96

    #mdlstm_1 parameters:
    mdlstm_1 = {'hidden_dim': 2,
                'context_window_size': (4,2), 
                'scope':'mdlstm_1',
                'use_peepholes': True}

    #fc_1 parameters:
    fc_1 = {'hidden_dim': 6,
            'context_window_size': (4,1),
            'activation': tf.nn.tanh,
            'scope':'fc_1'}

    #mdlstm_2 parameters:
    mdlstm_2 = {'hidden_dim': 10,
                'context_window_size': (1,1), 
                'scope':'mdlstm_2',
                'use_peepholes': True}

    #fc_2 parameters:
    fc_2 = {'hidden_dim': 20,
            'context_window_size': (4,1),
            'activation': tf.nn.tanh, 
            'scope':'fc_2'}

    #mdlstm_3 parameters:
    mdlstm_3 = {'hidden_dim': 50,
                'context_window_size': (1,1),
                'scope':'mdlstm_3',
                'use_peepholes': True}

    arch = [('MDLSTM', mdlstm_1), ('FullyConnected', fc_1), ('MDLSTM', mdlstm_2), ('FullyConnected', fc_2), ('MDLSTM', mdlstm_3)]

class MDLSTM_CTC(Model):
    def build_graph(self):
        #include color channel and make sure that X is shape [batch_size, height, width, channels]
        output = tf.expand_dims(tf.transpose(self.X_placeholder, [0, 2, 1]), axis=3)
        seq_len = self.seq_len_placeholder

        for layer, layer_config in self.config.arch:
            output, seq_len = self.define_layer(output, layer, layer_config, seq_len)

        output = slim.fully_connected(self.collapse(output), self.vocab_size+1, activation_fn=None)
        
        variable_summaries(output, 'outputs')
        self.max_seq_len = int(output.get_shape()[1])
                        
        self.loss, decoded = self.CTC(output, self.y_placeholder, seq_len)
        self.decoded_train = self.decoded_infer = decoded
        self.train_op, self.grad_norm = self.optimize(self.loss)

    def define_layer(self, X, layer, layer_config, seq_len):
        hidden_dim, context_window_size, scope = layer_config['hidden_dim'], layer_config['context_window_size'], layer_config['scope']
        if context_window_size:
            X = reshape_using_context_window(X, context_window_size)
            if self.config.use_dynamic_lengths:
                seq_len = tf.ceil(seq_len/context_window_size[1])
                max_time = X.get_shape().as_list()[2]
                seq_len = tf.cast(tf.clip_by_value(seq_len, 0, max_time), tf.int32)
        if layer == 'MDLSTM':
            cell = MDLSTMCell(hidden_dim, use_peepholes=layer_config['use_peepholes'])
            out = multidimensional_dynamic_rnn(cell, X, dtype=X.dtype, time_major=False, 
                    sequence_length=seq_len, scope=scope)
        
        elif layer == 'FullyConnected':
            out = slim.fully_connected(X, hidden_dim, activation_fn=None, scope=scope)
            out = layer_config['activation'](out)
        
        else:
            raise AssertionError('Layer %s not defined' % layer)

        return out, seq_len
    
    def collapse(self, X):
        collapse =  tf.reduce_sum(X, axis=1)
        
        return collapse


