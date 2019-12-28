from __future__ import print_function

import tensorflow as tf

from models.model_class import Model
from models.helpers import encoder_params_helper, variable_summaries
from models.cnn import *
from models.encoder import *

class CNN_RNN_CTC(Model):
    def _build_graph(self):
        if self.config.use_dynamic_lengths:
            X_seq_len = self.X_seq_len_placeholder
        else:
            X_seq_len = None

        if self.config.dropout is not None:
            dropout = self.dropout_placeholder
        else:
            dropout = None

        cnn_out, X_seq_len = self._setup_cnn(self.X_placeholder, X_seq_len, self.is_training_placeholder)

        if not self.config.use_dynamic_lengths:
            X_seq_len = tf.fill([tf.shape(cnn_out)[0]], tf.shape(cnn_out)[1])

        rnn_out = self._setup_rnn(cnn_out, X_seq_len, dropout, self.vocab_size)

        self.loss, decoded = self._setup_CTC(rnn_out, self.y_placeholder, X_seq_len, self.vocab_size) 
        self.decoded_train = self.decoded_infer = decoded
        self.train_op, self.grad_norm = self._optimize(self.loss)

        if self.config.verbose:
            img = tf.expand_dims(tf.transpose(self.X_placeholder, [0,2,1]), -1)
            tf.summary.image("inputs", img , max_outputs=self.config.max_outputs)

    def _setup_cnn(self, X, X_seq_len, is_training):
        X_expanded = tf.expand_dims(X, axis=3)

        with tf.variable_scope("cnn"):
            cnn = CNN(self.config)
            output, new_seq_len = cnn(X_expanded, X_seq_len, is_training)

        output_squeezed = tf.squeeze(output, axis=2)

        return output_squeezed, new_seq_len

    def _setup_rnn(self, X, X_seq_len, dropout, vocab_size):
        config = self.config
        params = encoder_params_helper(config.rnn_num_layers,
                                       config.rnn_unit_type,
                                       config.rnn_type,
                                       config.rnn_num_units,
                                       config.rnn_num_residual_layers,
                                       config.verbose)

        with tf.variable_scope("rnn"):
            encoder = Encoder(params)
            outputs, output_state = encoder(X, X_seq_len, dropout)

            output_layer = tf.layers.Dense(vocab_size+1)
            outputs = output_layer(outputs)

            if config.verbose:
                variable_summaries(output_layer.trainable_weights[0], 'output_layer_weights')
                variable_summaries(output_layer.trainable_weights[1], 'output_layer_biases')
                variable_summaries(outputs, 'linear_projections')

        return outputs
