import tensorflow as tf

from models.helpers import create_rnn_cell
from models.helpers import create_attention_images_summary
from models.helpers import get_label
from models.helpers import get_trainable_weights
from models.helpers import get_trainable_variables
from models.helpers import variable_summaries

__all__ = ["Decoder"]

class Decoder():
    """
    This class implements a decoder, with an attention option, for sequence-to-sequence
    models. The decoder is essentially a multi-recurrent neural network.
    """
    def __init__(self, params):
        """
        Args:
            params: hyperparameters of the decoder
        """
        self.params = params

    def __call__(self, X, X_seq_len, X_state, y, y_seq_len, embedding_matrix, dropout, extra_codes, vocab_size, max_len, global_step): 
        """
        Args:
            X: input sequences - usually the outputs of an encoder - shaped [batch_size, time_steps, features]
            X_seq_len - lengths of input sequences shaped [batch_size]
            X_state: initial state of the decoder - usually the final state of an encoder - either a tensor shaped 
                    [batch_size, state_size] or a list of tensors of this shape
            y: target sequences shaped [batch_size, max(y_seq_len), embed_size]
            y_seq_len: lengths of target sequences shaped [batch_size]
            embedding_matrix: lookup table shaped [vocab_size, embed_size]
            dropout: dropout (1-keep_prob) to apply to the decoder cell
            extra_codes: dictionary containin index numbers for <SOS>, <EOS>, <pad>
            vocab_size: size of the vocabulary
            max_len: the maximum length that the predicted sequences can have in infer mode
            global_step: the number of steps that the model has taken

        Returns:
            train_output: logits during training mode shaped [batch_size, max(y_seq_len), vocab_size]
            train_alignment: the training alignment matrix shaped [max(y_seq_len), batch_size, time_steps]
            infer_output: predicted sequences shaped [batch_size, infer_len] where infer_len <= max_len
            infer_alignment: the inference alignment matrix shaped [infer_len, batch_size, time_steps]
        """
        params = self.params
        with tf.variable_scope("decoder"):
            cell = create_rnn_cell(params.decoder_unit_type,
                                   params.decoder_num_layers,
                                   params.decoder_num_units,
                                   dropout=dropout,
                                   base_name=None,
                                   num_residual_layers=params.decoder_num_residual_layers,
                                   residual_fn=None)

            output_layer = tf.layers.Dense(vocab_size, name="output_layer")

            train_output, train_alignment = self._decoder_train(cell,
                                                                X,
                                                                X_seq_len,
                                                                X_state, 
                                                                y,
                                                                y_seq_len,
                                                                embedding_matrix,
                                                                output_layer,
                                                                global_step)

            if params.verbose:
                variable_summaries(output_layer.trainable_weights[0], "final_layer_weights")
                variable_summaries(output_layer.trainable_weights[1], "final_layer_biases")
                create_attention_images_summary(train_alignment, "train_attention", max_outputs=params.max_outputs)
                #create_attention_images_summary(infer_alignment, "infer_attention", max_outputs=params.max_outputs)

        with tf.variable_scope("decoder", reuse=True):
            infer_output, infer_alignment = self._decoder_infer(cell,
                                                                X,
                                                                X_seq_len,
                                                                X_state,
                                                                embedding_matrix,
                                                                output_layer,
                                                                extra_codes,
                                                                max_len)

        return train_output, train_alignment, infer_output, infer_alignment

    def _decoder_train(self, cell, X, X_seq_len, X_state, y, y_seq_len, embedding_matrix, output_layer, global_step):
        """
        Args:
            cell: the decoder cell - a RNN instance
            See __call__ function for other args

        Returns (outputs, alignments):
            outputs: logits shaped [batch_size, max(y_seq_len), vocab_size]
            alignments: the alignment_history shaped [batch_size, max_len(y_seq_len), time_steps] if
                        params.use_attention = True, otherwise None

        Raises:
            ValueError: if both pass_hidden_state and use_attention are False
        """
        params = self.params

        if params.do_scheduled_sampling:
            helper = self._get_scheduled_sampling_helper(y, y_seq_len, embedding_matrix, global_step)
        else:
            helper = tf.contrib.seq2seq.TrainingHelper(y, y_seq_len)

        if params.use_attention:
            cell, decoder_initial_state = self._wrap_attention(cell, X, X_seq_len, X_state, alignment_history=True)

        else:
            if not params.pass_hidden_state:
                raise ValueError("pass_hidden_state must be set True for models not using attention")
            decoder_initial_state = X_state

        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state, output_layer=output_layer)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope="decoder")

        if params.verbose:
            for w, v in zip(get_trainable_weights(cell), get_trainable_variables(cell)):
                variable_summaries(w, "{}".format(v))

        return outputs.rnn_output, final_context_state.alignment_history.stack()

    def _decoder_infer(self, cell, X, X_seq_len, X_state, embedding_matrix, output_layer, extra_codes, max_len):
        """
        Args:
            cell: the decoder cell - a RNN instance
            See __call__ function for other args

        Returns (predicted_sequences, alignments):
            predicted_sequences: the predicted sequences
            alignments: the alignment_history shaped [batch_size, infer_len, time_steps] where infer_len
                        < max_len if params.use_attention = True and params.decoder_type = "greedy_search",
                        otherwise None

        Raises:
            ValueError: if both pass_hidden_state and use_attention are False or if params.decoder_type is
                        not known
        """
        params = self.params
        if not params.use_attention and not params.pass_hidden_state:
            raise ValueError("pass_hidden_state must be set True for models not using attention")

        start_tokens = tf.fill([tf.shape(X)[0]], extra_codes["<SOS>"])

        if params.decoder_type == "greedy_search":
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_matrix,
                                                              start_tokens,
                                                              extra_codes["<EOS>"])
            if params.use_attention:
                cell, decoder_initial_state = self._wrap_attention(cell, X, X_seq_len, X_state, alignment_history=True)
            else:
                decoder_initial_state = X_state

            decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state, output_layer=output_layer)

        elif params.decoder_type == "beam_search":
            if params.use_attention:
                tiled_X, tiled_X_seq_len, tiled_X_state = self._prepare_beam_search_decoder_inputs(X, X_seq_len, X_state)
                cell, decoder_initial_state = self._wrap_attention(cell, tiled_X, tiled_X_seq_len, tiled_X_state, alignment_history=False)
            else:
                decoder_initial_state = X_state

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell,
                                                           embedding_matrix,
                                                           start_tokens,
                                                           extra_codes["<EOS>"],
                                                           decoder_initial_state,
                                                           params.beam_width,
                                                           output_layer=output_layer,
                                                           length_penalty_weight=params.length_penalty_weight,
                                                           coverage_penalty_weight=params.coverage_penalty_weight)

        else:
            raise ValueError("Unknown decoder_type %s" % params.decoder_type)

        outputs, final_context_states, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_len, scope="decoder")

        if params.decoder_type == "greedy_search":
            return outputs.sample_id, final_context_states.alignment_history.stack()
        elif params.decoder_type == "beam_search":
            return outputs.predicted_ids[:,:,0], None

    def _wrap_attention(self, cell, memory, memory_seq_len, initial_state, alignment_history=False):
        """
        Args:
            cell: the decoder cell to wrap attention around, a RNN instance
            memory: the memory to consult when decoding shaped [batch_size, time_steps, features]
            memory_seq_len: the memory sequence length shaped [batch_size]
            initial_state: initial state of the decoder shaped [batch_size, state_size]
            alignment_history: whether to store alignments or not

        Returns:
            cell: RNN cell wrapped with a AttentionWrapper
            decoder_initial_state: the initial state of the decoder
        """ 
        params = self.params

        attention_mechanism = _create_attention_mechanism(params.attention_type, params.attention_num_units, memory, memory_seq_len)

        # Wrap cell with attention wrapper
        cell = tf.contrib.seq2seq.AttentionWrapper(cell,
                                                   attention_mechanism,
                                                   attention_layer_size=params.attention_num_units,
                                                   alignment_history=alignment_history,
                                                   output_attention=params.output_attention,
                                                   name="attention")

        batch_size = tf.shape(memory)[0]
        dtype = memory.dtype
        if params.pass_hidden_state:
            # Need to pass named arguments to clone(). Otherwise it raises an error.
            decoder_initial_state = cell.zero_state(batch_size, dtype).clone(cell_state=initial_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size, dtype)

        return cell, decoder_initial_state

    def _prepare_beam_search_decoder_inputs(self, X, X_seq_len, X_state):
        """
        Tiles inputs. See https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BeamSearchDecoder for why
        this is needed.
        """
        multiplier = self.params.beam_width

        return _tile_sequence(X, multiplier), _tile_sequence(X_seq_len, multiplier), _tile_sequence(X_state, multiplier)

    def _get_scheduled_sampling_helper(self, y, y_seq_len, embedding_matrix, global_step):
        """
        Sets up the scheduled sampling helper. See: https://arxiv.org/pdf/1506.03099v3.pdf

        Args:
            y: target sequences shaped [batch_size, max(y_seq_len), embed_size]
            y_seq_len: sequence lengths of the target sequences shaped [batch_size]
            embedding_matrix: contains embeddings for all elements in the vocabulary
            global_step: the number of steps that the model has taken

        Returns:
            the scheduled sampling helper
        """
        params = self.params

        if params.anneal_not_sampling_prob:
            not_sampling_prob = tf.train.exponential_decay(params.initial_not_sampling_prob,
                                                           global_step,
                                                           params.anneal_not_sampling_prob_every,
                                                           params.anneal_not_sampling_prob_rate)
        else:
            not_sampling_prob = params.initial_not_sampling_prob

        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(y,
                                                                     y_seq_len,
                                                                     embedding_matrix,
                                                                     1-not_sampling_prob)

        return helper

def _tile_sequence(to_tile, multiplier):
    """
    Copies the tensor to_tile 'multiplier' times and concatenates all copies along axis 0 (batch_axis)
    """
    return tf.contrib.seq2seq.tile_batch(to_tile, multiplier)

def _create_attention_mechanism(attention_type, num_units, memory, memory_sequence_length):
    """
    Create attention mechansim based on attention_type

    Args:
        attention_type: type of attention to use
        num_units: the size of the attention layer
        memory: the memory to consult with
        memory_sequence_length: sequence lengths for the memory

    Returns:
        attention_mechanism: the attention layer

    Raises:
        ValueError: if attention_type is not known
    """

    if attention_type == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units,
                                                                memory,
                                                                memory_sequence_length=memory_sequence_length)
    elif attention_type == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units,
                                                                memory,
                                                                memory_sequence_length=memory_sequence_length,
                                                                scale=True)
    elif attention_type == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units,
                                                                   memory,
                                                                   memory_sequence_length=memory_sequence_length)
    elif attention_type == "normed_bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units,
                                                                   memory,
                                                                   memory_sequence_length=memory_sequence_length,
                                                                   normalize=True)
    else:
        raise ValueError("Unknown attention_type %s " % attention_type)

    return attention_mechanism
