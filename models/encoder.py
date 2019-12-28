import tensorflow as tf

from models.helpers import create_rnn_cell
from models.helpers import create_attention_images_summary
from models.helpers import get_label
from models.helpers import get_trainable_weights
from models.helpers import variable_summaries

__all__ = ["Encoder"]

class Encoder():
    """
    This class implements a multi-recurrent neural network as an encoder
    for a sequence-to-sequence model.
    """
    def __init__(self, params):
        """
        Args:
            params: hyperparameters of the encoder
        """
        self.params = params

    def __call__(self, X, X_seq_len, dropout):
        """
        Args:
            X: input sequences shaped [batch_size, time_steps, features]
            X: input sequence lenghts shaped [batch_size]
            dropout: dropout (1-keep_prob) to apply to the encoder cell

        Returns:
            encoder_outputs: encoder outputs shaped:
                - [batch_size, time_steps, params.encoder_num_units] if params.encoder_type = "uni"
                - [batch_size, time_steps, 2*params.encoder_num_units] if params.encoder_type = "bi"
            encoder_state: final state of the encodder which is either a
                - tensor shaped [batch_size, params.encoder_num_units] if params.encoder_num_layers = 1 and
                  params.encoder_type = "uni"
                - list of tensors shaped [batch_size, params.encoder_num_units] of size params.encoder_num_layers
        
        Raises:
            ValueError if params.encoder_num_layers and params.encoder_num_residual_layers are not a multiple
            of 2 for params.encoder_type = "bi" (note that the latter may be zero, however)
        """
        params = self.params

        if params.encoder_type == "uni":
            encoder_outputs, encoder_state = _build_unidirectional_encoder(params, X, X_seq_len, dropout)
        
        elif params.encoder_type == "bi":
            if params.encoder_num_residual_layers % 2 != 0:
                raise ValueError("Residual layers must be even for bidirectional encoder")
            if params.encoder_num_layers % 2 != 0:
                raise ValueError("Number of layers must be even for bidirectional encoder")

            encoder_outputs, encoder_state = _build_bidirectional_encoder(params, X, X_seq_len, dropout)
        
        else:
            raise ValueError("Unknown encoder type %s" % params.encoder_type)

        variable_summaries(encoder_outputs, 'outputs')

        return encoder_outputs, encoder_state

def _build_unidirectional_encoder(params, X, X_seq_len, dropout):
    """
    Args:
        params: params of the encoder
        X: input sequences
        X_seq_len: input sequence lengths
        dropout: dropout to apply to encoder cell
    
    Returns:
        outputs: encoder outputs shaped [batch_size, time_steps, params.encoder_num_units]
        output_state: final state of the encoder which is either a
            - tensor shaped [batch_size, params.encoder_num_units] if params.encoder_num_layers = 1
            - list of tensors shaped [batch_size, params.encoder_num_units] of size params.encoder_num_layers
    """
    cell = create_rnn_cell(params.encoder_unit_type,
                           params.encoder_num_layers,
                           params.encoder_num_units,
                           dropout=dropout,
                           base_name=None,
                           num_residual_layers=params.encoder_num_residual_layers,
                           residual_fn=None)

    outputs, output_state = tf.nn.dynamic_rnn(cell,
                                              X,
                                              sequence_length=X_seq_len,
                                              dtype=X.dtype,
                                              time_major=False)

    if params.verbose:
        for i, w in enumerate(get_trainable_weights(cell)):
            variable_summaries(w, "cell_{}_{}".format(get_label(i), i))

    return outputs, output_state

def _build_bidirectional_encoder(params, X, X_seq_len, dropout):
    """
    Args:
        params: params of the encoder
        X: input sequences shaped [batch_size, time_steps, features]
        X: input sequence lenghts shaped [batch_size]
        dropout: dropout (1-keep_prob) to apply to the encoder cell

    Returns:
        outputs: encoder outputs shaped [batch_size, time_steps, 2*params.encoder_num_units]
        encoder_state: final state of the encoder which is a list of tensors shaped [batch_size, params.encoder_num_units]
        of size params.encoder_num_layers

    This is taken from https://github.com/tensorflow/nmt/blob/master/nmt/model.py

    An alternative approach (for the case when params.encoder_num_layers > 1):
        Halve params.encoder_num_units instead of params.encoder_num_layers and create params.encoder_num_layers
        forward and backward cells. Then use the tf.contrib.rnn.stack_bidirectional_dynamic_rnn function to stack
        multiple bidirectional rnns on top of one another. Finally, concat the final forward and backward states
        for each layer. Note that in this case, the enocder outputs will be of shape [batch_size, time_steps,
        params.encoder_num_units] while the encoder final states will be a list sized params.encoder_num_layers
        containing tensors of shape [batch_size, time_steps, params.encoder_num_units].

    Yet another approach:
        Instead of using the tf.contrib.rnn.stack_bidirectional_dynamic_rnn function as in the above approach, one
        may also use tf.nn.bidirectional_dynamic_rnn as below (while halving params.encoder_num_units instead of
        params.encoder_num_layers) and then concating the encoder final states as follows:

            def _concat(state_1, state_2):
                assert type(state_1) == type(state_2)
                if type(state_1) == LSTMStateTuple:
                    return LSTMStateTuple(tf.concat([state_1[0], state_2[0]], -1), tf.concat([state_1[1], state_2[1]], -1))
                else:
                    return tf.concat([state_1, state_2], -1)

            _output_state = []
            for layer_id in range(self.config.encoder_num_layers):
                _output_state.append(_concat(output_states[0][layer_id], output_states[1][layer_id]))
            output_state = tuple(_output_state)

        Note that in this case the outputs will also have the same structure as in the alternative approach above.
    """

    num_bi_residual_layers = int(params.encoder_num_residual_layers/2)
    num_bi_layers = int(params.encoder_num_layers/2)

    cell_fw = create_rnn_cell(params.encoder_unit_type,
                              num_bi_layers,
                              params.encoder_num_units,
                              dropout=dropout,
                              base_name=None,
                              num_residual_layers=num_bi_residual_layers,
                              residual_fn=None)

    cell_bw = create_rnn_cell(params.encoder_unit_type,
                              num_bi_layers,
                              params.encoder_num_units,
                              dropout=dropout,
                              base_name=None,
                              num_residual_layers=num_bi_residual_layers,
                              residual_fn=None)

    outputs, output_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                            cell_bw,
                                                            X,
                                                            sequence_length=X_seq_len,
                                                            dtype=X.dtype,
                                                            time_major=False)
    outputs = tf.concat(outputs, 2)

    if num_bi_layers == 1:
        encoder_state = output_state
    # Alternatively concat forward and backward states
    else:
        encoder_state = []
        for layer_id in range(num_bi_layers):
            encoder_state.append(output_state[0][layer_id])     # forward
            encoder_state.append(output_state[1][layer_id])     # backward
        encoder_state = tuple(encoder_state)

    if params.verbose:
        for i, (w_fw, w_bw) in enumerate(zip(get_trainable_weights(cell_fw), get_trainable_weights(cell_bw))):
            variable_summaries(w_fw, "cell_fw_{}_{}".format(get_label(i), i))
            variable_summaries(w_bw, "cell_bw_{}_{}".format(get_label(i), i))

    return outputs, encoder_state
