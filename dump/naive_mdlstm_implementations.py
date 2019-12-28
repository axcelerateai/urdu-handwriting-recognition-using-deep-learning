class NaiveMDLSTM():
    """Naive Implementations"""
    def naive_initialize_cell(self):
        D = self.params['feature_vector_length']
        H = self.params['hidden_dim']

        D_sqrt = tf.sqrt(tf.cast(D,tf.float32))
        H_sqrt = tf.sqrt(tf.cast(H,tf.float32))

        with tf.variable_scope(self.scope):
            self.weights = {}

            """ Input weights """
            self.weights['Wxi'] = tf.get_variable('Wxi', shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
            for d in range(2):
                self.weights['Wxf_'+str(d+1)] = tf.get_variable('Wxf_'+str(d+1), shape=[D, H], 
                        initializer=tf.contrib.layers.xavier_initializer())
            self.weights['Wxg'] = tf.get_variable('Wxg', shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
            self.weights['Wxo'] = tf.get_variable('Wxo', shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())

            """ Hidden weights """
            for d in range(2):
                self.weights['Whi_'+str(d+1)] = tf.get_variable('Whi_'+str(d+1), shape=[H, H], 
                        initializer=tf.contrib.layers.xavier_initializer())
            for d1 in range(2):
                for d2 in range(2):
                    self.weights['Whf_'+str(d1+1)+'_'+str(d2+1)] = tf.get_variable('Whf_'+str(d1+1)+'_'+str(d2+1), shape=[H, H], 
                            initializer=tf.contrib.layers.xavier_initializer())
            for d in range(2):
                self.weights['Whg_'+str(d+1)] = tf.get_variable('Whg_'+str(d+1), shape=[H, H], 
                        initializer=tf.contrib.layers.xavier_initializer())
            for d in range(2):
                self.weights['Who_'+str(d+1)] = tf.get_variable('Who_'+str(d+1), shape=[H, H], 
                        initializer=tf.contrib.layers.xavier_initializer())

            """ Cell weights """
            self.weights['Wci'] = tf.get_variable('Wci', shape=[H, H], initializer=tf.contrib.layers.xavier_initializer())
            for d in range(2):
                self.weights['Wcf_'+str(d+1)] = tf.get_variable('Wcf_'+str(d+1), shape=[H, H], 
                        initializer=tf.contrib.layers.xavier_initializer())
            self.weights['Wco'] = tf.get_variable('Wco', shape=[H, H], initializer=tf.contrib.layers.xavier_initializer())

    def naive_step_forward(self, features, prev_hidden_states, prev_cell_states):
        """ features: batch_size X feature_length 
            prev_hidden_states: tuple of two hidden states, one for each dimension
            prev_cell_states: tuple of two cell states, one for each dimension """

        H = self.params['hidden_dim']

        """" Input Gate, i """
        i = tf.matmul(features,self.weights['Wxi'])
        W2 = self.weights['Wci']
        for d in range(2): 
            W1 = self.weights['Whi_'+str(d+1)]
            i += tf.matmul(prev_hidden_states[d], W1) + tf.matmul(prev_cell_states[d], W2)
        #i += self.weights['b_i']
        i = layer_normalization(i, scope='i/')
        i = tf.nn.sigmoid(i)

        """ Forget Gate, f """
        f = []
        for d1 in range(2):
            tmp = tf.matmul(features,self.weights['Wxf_'+str(d1+1)])
            for d2 in range(2):
                tmp += tf.matmul(prev_hidden_states[d2], self.weights['Whf_'+str(d1+1)+'_'+str(d2+1)])
            tmp += tf.matmul(prev_cell_states[d1], self.weights['Wcf_'+str(d1+1)])
            #tmp += self.weights['b_f_'+str(d1+1)]
            tmp = layer_normalization(tmp, scope='f'+str(d1)+'/')
            tmp = tf.nn.sigmoid(tmp) 
            f.append(tmp)

        """ Cell Gate, g """
        g = tf.matmul(features, self.weights['Wxg'])
        for d in range(2):
            g += tf.matmul(prev_hidden_states[d], self.weights['Whg_'+str(d+1)])
        #g += self.weights['b_g']
        g = layer_normalization(g, scope='g/')
        g = tf.nn.tanh(g)

        """ Next Cell State """
        next_cell_state = i*g
        for d in range(2):
            next_cell_state += f[d]*prev_cell_states[d]
            
        """ Output Gate, o """
        o = tf.matmul(features, self.weights['Wxo'])
        for d in range(2):
            o += tf.matmul(prev_hidden_states[d], self.weights['Who_'+str(d+1)])
        o += tf.matmul(next_cell_state, self.weights['Wco'])
        #o += self.weights['b_o']
        o = layer_normalization(o, scope='o/')
        o = tf.nn.sigmoid(o)
        
        """ Next Hidden State """
        next_hidden_state = o*tf.nn.tanh(layer_normalization(next_cell_state, scope='cell/'))

        return next_hidden_state, LSTMStateTuple(next_hidden_state, next_cell_state)
    
    def naive_forward(self, X):
        """X: batch_size X image_height X image_width X feature_length """
        
        _, HH, WW, F = X.get_shape().as_list()
        H = self.params['hidden_dim']
         
        dims = tf.stack([tf.shape(X)[0], H])
        h = [[0 for x in range(WW)] for y in range(HH)]
        c = [[0 for x in range(WW)] for y in range(HH)]
        for i in range(HH):
            for j in range(WW):
                if i>0 and j>0:
                    prev_hidden_states = (h[i][j-1], h[i-1][j])
                    prev_cell_states = (c[i][j-1], c[i-1][j])
                elif i==0 and j>0:
                    prev_hidden_states = (h[i][j-1], tf.fill(dims, 0.0))
                    prev_cell_states = (c[i][j-1], tf.fill(dims, 0.0))
                elif i>0 and j==0:
                    prev_hidden_states = (tf.fill(dims, 0.0), h[i-1][j])
                    prev_cell_states = (tf.fill(dims, 0.0), c[i-1][j])
                else:
                    prev_hidden_states = (tf.fill(dims, 0.0), tf.fill(dims, 0.0))
                    prev_cell_states = (tf.fill(dims, 0.0), tf.fill(dims, 0.0))
                
                h[i][j], c[i][j] = self.step_forward(X[:,i,j,:], prev_hidden_states, prev_cell_states)
 
        out = []
        for i in range(HH):
            tmp = tf.expand_dims(h[i][0], 0)
            for j in range(1,WW):
                tmp = tf.concat([tmp, tf.expand_dims(h[i][j], 0)], 0)
            out.append(tmp)
        
        out = tf.stack(out, 0)
        out = tf.transpose(out, [2,0,1,3])
        
        return out

class MDLSTM():
    def __init__(self, hidden_dim, input_dim, context_window_size, scope=None, use_peepholes=True):
        self.params = {}
        self.params['hidden_dim'] = hidden_dim
        self.params['input_dim'] = input_dim
        self.params['context_window_size'] = context_window_size
        self.params['feature_vector_length'] = context_window_size[0]*context_window_size[1]*input_dim
        self.use_peepholes = use_peepholes
        if not isinstance(scope, str):
            scope = ''
        self.scope = scope
        self.initialize_cell()
   
    def initialize_cell(self):
        D = self.params['feature_vector_length']
        H = self.params['hidden_dim']
        self.weights = {}
        
        with tf.variable_scope(self.scope):
            self.weights['W_1'] = tf.get_variable('W_1', shape=[2*H+D, 5*H], initializer=tf.contrib.layers.xavier_initializer())
            self.weights['b'] = tf.get_variable('b', shape=[5*H], initializer=tf.constant_initializer)
            
            if self.use_peepholes:
                self.weights['W_2'] = tf.get_variable('W_2', shape=[2*H, H], initializer=tf.contrib.layers.xavier_initializer())
                self.weights['W_3'] = tf.get_variable('W_3', shape=[H, H], initializer=tf.contrib.layers.xavier_initializer())
                self.weights['W_4'] = tf.get_variable('W_4', shape=[H, H], initializer=tf.contrib.layers.xavier_initializer())
                self.weights['W_5'] = tf.get_variable('W_5', shape=[H, H], initializer=tf.contrib.layers.xavier_initializer())

    def step_forward(self, features, prev_hidden_states, prev_cell_states):
        H = self.params['hidden_dim']
        h1, h2 = prev_hidden_states
        c1, c2 = prev_cell_states
        
        concat = tf.concat([features, h1, h2], axis=1)
        prod = tf.matmul(concat, self.weights['W_1']) + self.weights['b']
        
        i, f1, f2, o, g = tf.split(value=prod, num_or_size_splits=5, axis=1)
        
        if self.use_peepholes:
            i += tf.matmul(tf.concat([c1, c2], axis=1), self.weights['W_2'])
            f1 += tf.matmul(c1, self.weights['W_3'])
            f2 += tf.matmul(c2, self.weights['W_4'])
        
        i = tf.nn.sigmoid(layer_normalization(i, scope='i/'))
        f1 = tf.nn.sigmoid(layer_normalization(f1, scope='f1/'))
        f2 = tf.nn.sigmoid(layer_normalization(f2, scope='f2/'))
        g = tf.nn.tanh(layer_normalization(g, scope='g/'))

        next_cell_state = f1*c1 + f2*c2 + i*g
        
        if self.use_peepholes:
            o += slim.fully_connected(next_cell_state, H, activation_fn=None) 
            o += tf.matmul(next_cell_state, self.weights['W_5'])
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
            states = states.write(H*W, LSTMStateTuple(tf.zeros([N, self.params['hidden_dim']], tf.float32),
                                                      tf.zeros([N, self.params['hidden_dim']], tf.float32)))

            """ define counter """
            t = tf.constant(0)

            """ define operations at each time step """
            def body(t_, outputs_, states_):
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

            outputs = tf.transpose(tf.reshape(outputs, [H, W, -1, self.params['hidden_dim']]), [2,0,1,3])

        return outputs, states

    def check_forward(self, inputs, prev_hidden_states , prev_cell_states):
        c1, c2 = prev_cell_states
        h1, h2 = prev_hidden_states

        # change bias argument to False since LN will add bias via shift
        concat = _linear([inputs, h1, h2], 5*self.params['hidden_dim'], False)

        i, f1, f2, g, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

        # add layer normalization to each gate
        i = layer_normalization(i, scope='i/')
        f1 = layer_normalization(f1, scope='f1/')
        f2 = layer_normalization(f2, scope='f2/')
        g = layer_normalization(f1, scope='g/') 
        o = layer_normalization(o, scope='o/')
        
        new_c = (c1 * tf.nn.sigmoid(f1) + 
                c2 * tf.nn.sigmoid(f2) + tf.nn.sigmoid(i) * tf.nn.tanh(g))

        # add layer_normalization in calculation of new hidden state
        new_h = tf.nn.tanh(layer_normalization(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
        new_state = LSTMStateTuple(new_c, new_h)

        return new_h, new_state

class FullyConnectedWithContextWindow():
    """ Implements a fully connected layer with context window """
    def __init__(self, hidden_dim, input_dim, context_window_size, activation, scope=None):
        self.params = {}
        self.params['hidden_dim'] = hidden_dim
        self.params['input_dim'] = input_dim
        self.params['context_window_size'] = context_window_size
        self.params['feature_vector_length'] = context_window_size[0]*context_window_size[1]*input_dim
        self.params['activation'] = activation
        if not isinstance(scope, str):
            scope = ''
        self.scope = scope
        self.initialize_cell()
    
    def initialize_cell(self):
        F = self.params['feature_vector_length']
        H = self.params['hidden_dim']
        
        with tf.variable_scope(self.scope):
            self.weights = {}
            self.weights['W'] = tf.get_variable('W', shape=[F, H], initializer=tf.contrib.layers.xavier_initializer())
            self.weights['b'] = tf.get_variable('b', shape=[H], initializer=tf.constant_initializer) 
        
    def __call__(self, X):
        """ X: batch_size X height X width X depth """
        
        features = get_features_using_context_window(X, self.params['context_window_size'])
        _, H, W, D = features.get_shape()
        features = tf.reshape(features, [-1,D])
        out = tf.matmul(features, self.weights['W']) + self.weights['b']
        out = tf.reshape(out, [-1, H, W, self.params['hidden_dim']])

        return self.params['activation'](out)

