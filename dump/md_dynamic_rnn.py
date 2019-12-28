"""RNN helpers for TensorFlow models."""
#TODO: Remove unused imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn import *
from tensorflow.python.ops.rnn import _transpose_batch_time, _best_effort_input_batch_size, _infer_state_dtype, _maybe_tensor_shape_from_tensor, _rnn_step
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.contrib.rnn import LSTMStateTuple

# pylint: disable=protected-access
_concat = rnn_cell_impl._concat
# pylint: enable=protected-access
# pylint: disable=unused-argument

def dynamic_rnn(cell, inputs, width, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
  """Creates a recurrent neural network specified by RNNCell `cell`.

  Performs fully dynamic unrolling of `inputs`.

  Example:

  ```python
  # create a BasicRNNCell
  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

  # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

  # defining initial state
  initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

  # 'state' is a tensor of shape [batch_size, cell_state_size]
  outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                     initial_state=initial_state,
                                     dtype=tf.float32)
  ```

  ```python
  # create 2 LSTMCells
  rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

  # create a RNN cell composed sequentially of a number of RNNCells
  multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

  # 'outputs' is a tensor of shape [batch_size, max_time, 256]
  # 'state' is a N-tuple where N is the number of LSTMCells containing a
  # tf.contrib.rnn.LSTMStateTuple for each cell
  outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                     inputs=data,
                                     dtype=tf.float32)
  ```


  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such
        elements.
      If `time_major == True`, this must be a `Tensor` of shape:
        `[max_time, batch_size, ...]`, or a nested tuple of such
        elements.
      This may also be a (possibly nested) tuple of Tensors satisfying
      this property.  The first two dimensions must match across all the inputs,
      but otherwise the ranks and other shape components may differ.
      In this case, input to `cell` at each time-step will replicate the
      structure of these tuples, except for the time dimension (from which the
      time is taken).
      The input to `cell` at each time step will be a `Tensor` or (possibly
      nested) tuple of Tensors each with dimensions `[batch_size, ...]`.
    width: The width of the image.
    sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
      Used to copy-through state and zero-out outputs when past a batch
      element's sequence length.  So it's more for performance than correctness.
    initial_state: (optional) An initial state for the RNN.
      If `cell.state_size` is an integer, this must be
      a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
      If `cell.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors.
      If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
      If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
      Using `time_major = True` is a bit more efficient because it avoids
      transposes at the beginning and end of the RNN calculation.  However,
      most TensorFlow data is batch-major, so by default this function
      accepts input and emits output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A pair (outputs, state) where:

    outputs: The RNN output `Tensor`.

      If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.

      If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.

      Note, if `cell.output_size` is a (possibly nested) tuple of integers
      or `TensorShape` objects, then `outputs` will be a tuple having the
      same structure as `cell.output_size`, containing Tensors having shapes
      corresponding to the shape data in `cell.output_size`.

    state: The final state.  If `cell.state_size` is an int, this
      will be shaped `[batch_size, cell.state_size]`.  If it is a
      `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
      If it is a (possibly nested) tuple of ints or `TensorShape`, this will
      be a tuple having the corresponding shapes. If cells are `LSTMCells`
      `state` will be a tuple containing a `LSTMStateTuple` for each cell.

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """
  rnn_cell_impl.assert_like_rnncell("cell", cell)

  with vs.variable_scope(scope or "rnn") as varscope:
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    if not context.executing_eagerly():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    flat_input = nest.flatten(inputs)

    if not time_major:
      # (B,T,D) => (T,B,D)
      flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
      flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
      sequence_length = math_ops.to_int32(sequence_length)
      if sequence_length.get_shape().ndims not in (None, 1):
        raise ValueError(
            "sequence_length must be a vector of length batch_size, "
            "but saw shape: %s" % sequence_length.get_shape())
      sequence_length = array_ops.identity(  # Just to find it in the graph.
          sequence_length, name="sequence_length")

    batch_size = _best_effort_input_batch_size(flat_input)

    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If there is no initial_state, you must give a dtype.")
      state = cell.zero_state(batch_size, dtype)

    def _assert_has_shape(x, shape):
      x_shape = array_ops.shape(x)
      packed_shape = array_ops.stack(shape)
      return control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
          ["Expected shape for Tensor %s is " % x.name,
           packed_shape, " but saw shape: ", x_shape])

    if not context.executing_eagerly() and sequence_length is not None:
      # Perform some shape validation
      with ops.control_dependencies(
          [_assert_has_shape(sequence_length, [batch_size])]):
        sequence_length = array_ops.identity(
            sequence_length, name="CheckSeqLen")

    inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

    (outputs, final_state) = _dynamic_mdlstm_loop(
        cell,
        inputs,
        width,
        state,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        sequence_length=sequence_length,
        dtype=dtype)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
    if not time_major:
      # (T,B,D) => (B,T,D)
      outputs = nest.map_structure(_transpose_batch_time, outputs)

    return (outputs, final_state)


def _dynamic_mdlstm_loop(cell,
                         inputs,
                         width,
                         initial_state,
                         parallel_iterations,
                         swap_memory,
                         sequence_length=None,
                         dtype=None):
  """Internal implementation of Dynamic RNN.

  Args:
    cell: An instance of RNNCell.
    inputs: A `Tensor` of shape [time, batch_size, input_size], or a nested
      tuple of such elements.
    initial_state: A `Tensor` of shape `[batch_size, state_size]`, or if
      `cell.state_size` is a tuple, then this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    parallel_iterations: Positive Python int.
    swap_memory: A Python boolean
    sequence_length: (optional) An `int32` `Tensor` of shape [batch_size].
    dtype: (optional) Expected dtype of output. If not specified, inferred from
      initial_state.

  Returns:
    Tuple `(final_outputs, final_state)`.
    final_outputs:
      A `Tensor` of shape `[time, batch_size, cell.output_size]`.  If
      `cell.output_size` is a (possibly nested) tuple of ints or `TensorShape`
      objects, then this returns a (possibly nested) tuple of Tensors matching
      the corresponding shapes.
    final_state:
      A `Tensor`, or possibly nested tuple of Tensors, matching in length
      and shapes to `initial_state`.

  Raises:
    ValueError: If the input depth cannot be inferred via shape inference
      from the inputs.
  """
  state = initial_state
  assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

  state_size = cell.state_size

  flat_input = nest.flatten(inputs)
  flat_output_size = nest.flatten(cell.output_size)

  # Construct an initial output
  input_shape = array_ops.shape(flat_input[0])
  time_steps = input_shape[0]
  batch_size = _best_effort_input_batch_size(flat_input)

  inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                           for input_ in flat_input)

  const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

  for shape in inputs_got_shape:
    if not shape[2:].is_fully_defined():
      raise ValueError(
          "Input size (depth of inputs) must be accessible via shape inference,"
          " but saw value None.")
    got_time_steps = shape[0].value
    got_batch_size = shape[1].value
    if const_time_steps != got_time_steps:
      raise ValueError(
          "Time steps is not the same for all the elements in the input in a "
          "batch.")
    if const_batch_size != got_batch_size:
      raise ValueError(
          "Batch_size is not the same for all the elements in the input.")

  # Prepare dynamic conditional copying of state & output
  def _create_zero_arrays(size):
    size = _concat(batch_size, size)
    return array_ops.zeros(
        array_ops.stack(size), _infer_state_dtype(dtype, state))

  flat_zero_output = tuple(_create_zero_arrays(output)
                           for output in flat_output_size)
  zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                      flat_sequence=flat_zero_output)

  if sequence_length is not None:
    min_sequence_length = math_ops.reduce_min(sequence_length)
    max_sequence_length = math_ops.reduce_max(sequence_length)
  else:
    max_sequence_length = time_steps

  time = array_ops.constant(0, dtype=dtypes.int32, name="time")

  with ops.name_scope("dynamic_rnn") as scope:
    base_name = scope

  def _create_ta(name, element_shape, dtype):
    return tensor_array_ops.TensorArray(dtype=dtype,
                                        size=time_steps,
                                        element_shape=element_shape,
                                        tensor_array_name=base_name + name)

  in_graph_mode = not context.executing_eagerly()
  if in_graph_mode:
    output_ta = tuple(
        _create_ta(
            "output_%d" % i,
            element_shape=(tensor_shape.TensorShape([const_batch_size])
                           .concatenate(
                               _maybe_tensor_shape_from_tensor(out_size))),
            dtype=_infer_state_dtype(dtype, state))
        for i, out_size in enumerate(flat_output_size))
    input_ta = tuple(
        _create_ta(
            "input_%d" % i,
            element_shape=flat_input_i.shape[1:],
            dtype=flat_input_i.dtype)
        for i, flat_input_i in enumerate(flat_input))
    input_ta = tuple(ta.unstack(input_)
                     for ta, input_ in zip(input_ta, flat_input))
  else:
    output_ta = tuple([0 for _ in range(time_steps.numpy())]
                      for i in range(len(flat_output_size)))
    input_ta = flat_input

  def _time_step(time, output_ta_t, states):
    """Take a time step of the dynamic RNN.

    Args:
      time: int32 scalar Tensor.
      output_ta_t: List of `TensorArray`s that represent the output.
      state: nested tuple of vector tensors that represent the state.

    Returns:
      The tuple (time + 1, output_ta_t with updated flow, new_state).
    """
    if in_graph_mode:
      input_t = tuple(ta.read(time) for ta in input_ta)
      # Restore some shape information
      for input_, shape in zip(input_t, inputs_got_shape):
        input_.set_shape(shape[1:])
    else:
      input_t = tuple(ta[time.numpy()] for ta in input_ta)

    input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
    row_time = time % width
    states_1 = control_flow_ops.cond(math_ops.less(time, width),
            lambda: states.read(time_steps), 
            lambda: states.read(time - width))
    states_2 = control_flow_ops.cond(math_ops.equal(row_time, 0),
            lambda: states.read(time_steps), 
            lambda: states.read(time - 1))

    call_cell = lambda: cell(input_t, tuple((states_1, states_2)))
    if sequence_length is not None:
      (output, new_state) = _rnn_step(
          time=row_time,
          sequence_length=sequence_length,
          min_sequence_length=min_sequence_length,
          max_sequence_length=max_sequence_length,
          zero_output=zero_output,
          state=state,
          call_cell=call_cell,
          state_size=state_size,
          skip_conditionals=True)
    else:
      (output, new_state) = call_cell()

    # Pack state if using state tuples
    output = nest.flatten(output)
    if in_graph_mode:
      output_ta_t = tuple(
          ta.write(time, out) for ta, out in zip(output_ta_t, output))
      states = states.write(time, new_state)
    else:
      for ta, out in zip(output_ta_t, output):
        ta[time.numpy()] = out

    return (time + 1, output_ta_t, states)

  if in_graph_mode:
    # Make sure that we run at least 1 step, if necessary, to ensure
    # the TensorArrays pick up the dynamic shape.
    loop_bound = time_steps
    #loop_bound = math_ops.minimum(
    #    time_steps, math_ops.maximum(1, max_sequence_length))
  else:
    # Using max_sequence_length isn't currently supported in the Eager branch.
    loop_bound = time_steps
  
  states = tensor_array_ops.TensorArray(dtype=dtype, size=time_steps+1, clear_after_read=False)
  states = states.write(time_steps, cell.zero_state(batch_size, dtype))

  _, output_final_ta, final_state = control_flow_ops.while_loop(
      cond=lambda time, *_: time < loop_bound,
      body=_time_step,
      loop_vars=(time, output_ta, states),
      parallel_iterations=parallel_iterations,
      maximum_iterations=time_steps,
      swap_memory=swap_memory)

  # Unpack final output if not using output tuples.
  if in_graph_mode:
    final_outputs = tuple(ta.stack() for ta in output_final_ta)
    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
      shape = _concat(
          [const_time_steps, const_batch_size], output_size, static=True)
      output.set_shape(shape)
  else:
    final_outputs = output_final_ta

  final_outputs = nest.pack_sequence_as(
      structure=cell.output_size, flat_sequence=final_outputs)
  if not in_graph_mode:
    final_outputs = array_ops.stack(final_outputs, axis=0)

  return (final_outputs, final_state)

def multidimensional_dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
        dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None):
    
    ### inputs = [batch_size, height, width, channels]

    scope = scope or ""
    _, h, w, c = inputs.get_shape().as_list()
    inputs_tl = tf.reshape(inputs, [-1, h*w, c])
    inputs_bl = tf.reshape(tf.image.flip_up_down(inputs), [-1, h*w, c])
    inputs_tr = tf.reshape(tf.reverse_sequence(inputs, sequence_length, seq_axis=2, batch_axis=0), [-1, h*w, c])
    inputs_br = tf.reshape(tf.image.flip_up_down(inputs_tr), [-1, h*w, c])

    out_tl, state_tr = dynamic_rnn(cell, inputs_tl, w, sequence_length=sequence_length, initial_state=initial_state,
            dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major, scope=scope+"top_left")
    out_bl, state_bl = dynamic_rnn(cell, inputs_bl, w, sequence_length=sequence_length, initial_state=initial_state,
            dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major, scope=scope+"bottom_left")
    out_tr, state_tr = dynamic_rnn(cell, inputs_tr, w, sequence_length=sequence_length, initial_state=initial_state,
            dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major, scope=scope+"top_right")
    out_br, state_br = dynamic_rnn(cell, inputs_br, w, sequence_length=sequence_length, initial_state=initial_state,
            dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major, scope=scope+"bottom_right")
 
    out_tl = tf.reshape(out_tl, [-1, h, w, cell._num_units])
    out_bl_c = tf.reshape(out_bl, [-1, h, w, cell._num_units])
    out_tr = tf.reshape(out_tr, [-1, h, w, cell._num_units])
    out_br = tf.reshape(out_br, [-1, h, w, cell._num_units])

    concat = tf.concat([out_tl, out_bl_c, out_tr, out_br], 3)

    return concat
