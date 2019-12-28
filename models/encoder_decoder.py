from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.rnn import LSTMStateTuple
from models.model_class import Model
from models.helpers import *
from models.encoder import *
from models.decoder import *
from models.cnn import *

def _residual_fn(inputs, outputs):
    transform_layer = tf.layers.Dense(cell_size)    
    return transform_layer(inputs) + outputs

class Encoder_Decoder(Model):
    def _build_graph(self):
        features = self.X_placeholder
        X_seq_len = self.X_seq_len_placeholder if self.config.use_dynamic_lengths else None

        if self.config.extract_features:
            features, X_seq_len = self._setup_cnn(features, X_seq_len, self.is_training_placeholder)
        
        if not self.config.use_dynamic_lengths:
            X_seq_len = tf.fill([tf.shape(features)[0]], tf.shape(features)[1]) 

        if self.config.dropout is None:
            dropout = None
        else:
            dropout = self.dropout_placeholder

        encoder_output, encoder_final_state = self._setup_encoder(features, X_seq_len, dropout)
        embeddings, embedding_matrix = self._setup_embeddings(self.y_in_placeholder, self.vocab_size)

        self.decoded_train, self.train_alignment, self.decoded_infer, self.infer_alignment = self._setup_decoder(encoder_output,
                                                                                                                 X_seq_len,
                                                                                                                 encoder_final_state,
                                                                                                                 embeddings,
                                                                                                                 self.y_seq_len_placeholder,
                                                                                                                 embedding_matrix,
                                                                                                                 dropout,
                                                                                                                 self.extra_codes,
                                                                                                                 self.vocab_size,
                                                                                                                 self.y_max_len,
                                                                                                                 self.global_step)

        self.loss = self._setup_loss(self.decoded_train, self.y_out_placeholder, self.y_seq_len_placeholder)
        self.train_op, self.grad_norm = self._optimize(self.loss)

        if self.config.verbose:
            img = tf.expand_dims(tf.transpose(self.X_placeholder, [0,2,1]), -1)
            tf.summary.image("inputs", img , max_outputs=self.config.max_outputs)

    def _setup_cnn(self, X, X_seq_len, is_training):
        """
        Sets up the convolutional layers.
        
        Args:
            X: input sequence
            X_seq_len: inputs' sequence lengths
            is_training: whether the model is training, a boolean

        Returns: 
            outputs_squeezed: the layer's outputs shaped [batch_size, height, width]
            new_seq_len: shaped [batch_size] which are
                - either the new sequence lengths (after taking into account subsampling)
                  if config.use_dynamic_lengths is True
                - or the old sequence lengths
        """
        X_expanded = tf.expand_dims(X, axis=3)   
        
        with tf.variable_scope("cnn"):
            cnn = CNN(self.config)
            output, new_seq_len = cnn(X_expanded, X_seq_len, is_training)
            
        output_squeezed = tf.squeeze(output, axis=2)

        return output_squeezed, new_seq_len
    
    def _setup_encoder(self, X, X_seq_len, dropout):
        """
        Sets up the encoder.
        
        Args:
            X: input sequence
            X_seq_len: inputs' sequence lengths
            dropout: dropout (1-keep_prob) to apply to encoder cell

        Returns (see the Encoder class for more details): 
            outputs: encoder outputs
            output_state: final state of the encoder
        """
        with tf.variable_scope("encoder"):
            encoder = Encoder(self.config)
            outputs, output_state = encoder(X, X_seq_len, dropout)

        return outputs, output_state

    def _setup_embeddings(self, y, vocab_size):
        """
        Creates embeddings for all elements in the vocabulary and replaces each element in y with its embedding.

        Args:
            y: output labels
            vocab_size: vocabulary size

        Returns:
            embeddings: tensor of shape [batch_size, y.get_shape()[1], config.embed_size], the output embeddings
            embedding_matrix: the embedding matrix which contains embeddings of all elements in the vocabulary
        """
        with tf.variable_scope("embeddings"):
            embedding_matrix = tf.get_variable("embedding_matrix",
                                               shape=[vocab_size, self.config.embed_size],
                                               trainable=True)
            embeddings = tf.nn.embedding_lookup(embedding_matrix, y)
            variable_summaries(embedding_matrix, "embedding_matrix_weights")

        return embeddings, embedding_matrix

    def _setup_decoder(self, encoder_output, encoder_seq_len, encoder_final_state, y, y_seq_len, embedding_matrix, dropout, extra_codes,
                      vocab_size, max_len, global_step):
        """
        Sets up the decoder.

        Args:
            encoder_output: outputs of the encoder shaped [batch_size, time_steps, features]
            encoder_seq_len: sequence lengths of the encoder outputs shaped [batch_size]
            encoder_final_state: final state of the encoder and is either a tensor of shape [batch_size, encoder_num_units]
                                 or a list of tensors of this shape
            y: target sequences shaped [batch_size, max(y_seq_len), embed_size]
            y_seq_len: sequence lengths of the target sequence shaped [batch_size]
            embedding_matrix: lookup table for elements in the vocabulary shaped [vocab_size, embed_size]
            dropout: the dropout (1-keep_prob) to apply to the decoder cell
            extra_codes: a dictionary containing index numbers for <SOS>, <EOS>, <pad>
            vocab_size: the size of the vocabulary
            max_len: the maximum length that the predicted sequences can have in infer mode

        Returns:
            train_output: logits during training mode shaped [batch_size, max(y_seq_len), vocab_size]
            train_alignment: the training alignment matrix shaped [max(y_seq_len), batch_size, time_steps]
            infer_output: predicted sequences shaped [batch_size, infer_len] where infer_len <= max_len
            infer_alignment: the inference alignment matrix shaped [infer_len, batch_size, time_steps]
        """
        decoder = Decoder(self.config)
        train_output, train_alignment, infer_output, infer_alignment = decoder(encoder_output,
                                                                               encoder_seq_len,
                                                                               encoder_final_state,
                                                                               y,
                                                                               y_seq_len,
                                                                               embedding_matrix,
                                                                               dropout,
                                                                               extra_codes,
                                                                               vocab_size,
                                                                               max_len,
                                                                               global_step)

        return train_output, train_alignment, infer_output, infer_alignment

    def _setup_loss(self, logits, labels, labels_seq_len):
        """
        Computes cross entropy loss.

        Args:
            logits: the predictions shaped [batch_size, max(labels_seq_len), vocab_size]
            labels: the true labels shaped [batch_size, max(labels_seq_len)]
            labels_seq_len: the sequence lengths of the labels shaped [batch_size]

        Returns:
            loss: a scalar, the cross-entropy loss
        """
        mask = tf.sequence_mask(labels_seq_len, maxlen=None, dtype=tf.float32)
        loss = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(logits, labels, mask, average_across_timesteps=False))

        return loss
