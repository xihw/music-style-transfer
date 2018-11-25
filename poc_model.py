import tensorflow as tf

class SimpleLSTM():
    def __init__(self):
        self.input_size = 128
        self.hidden_unit = 256
        self.output_size = 128
        self.keep_prob = 0.8
    
    def prepare(self):
        # reset graph
        tf.reset_default_graph()
    
        # input output unit    
        self.inputs = tf.placeholder(tf.float32, (None, None, self.input_size))
        self.true_outputs = tf.placeholder(tf.float32, (None, None, self.output_size))
    
        # length of each piece of music
        self.seq_len = tf.placeholder(tf.int32, [None])
    
        # lstm cell with dropouts
        self.lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_unit)
        self.lstm_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_cell,
                                                       input_keep_prob=self.keep_prob, 
                                                       output_keep_prob=self.keep_prob)
        
        # link lstm cells
        self.lstm_out, self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell, 
                                                           self.inputs, 
                                                           sequence_length = self.seq_len,
                                                           dtype = tf.float32)

        # output layer
        self.W = tf.get_variable("W", shape=(self.output_size, self.hidden_unit), initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable("b", shape=(self.output_size, 1), initializer=tf.zeros_initializer())

        # reshape lstm out to [hidden_unit, None], prepare for matrix multiplication
        self.z = tf.transpose(tf.reshape(self.lstm_out, (-1, self.hidden_unit)))
        self.z = tf.matmul(self.W, self.z) + self.b

        # scale output from (0, 1) to (0, 127) in accordance with velocity range
        self.y_pred = tf.sigmoid(self.z) * 127

        # reshape true_outputs to the same dimension
        self.y_true = tf.transpose(tf.reshape(self.true_outputs, (-1, self.output_size)))

        # loss
        self.cost = tf.reduce_mean(tf.square(self.y_true - self.y_pred))
        
    
    def train_batch(self, X, Y, epochs=100):
        
        m = X.shape[0]
        Tx = X.shape[1]
        seq_len = [Tx] * m
        
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        
        
        self.sess = tf.Session()

        # Run the initialization
        self.sess.run(tf.global_variables_initializer())
        
        # Do the training loop
        for epoch in range(epochs):
            _, cost = self.sess.run([self.optimizer, self.cost], 
                                   feed_dict={
                                       self.inputs : X,
                                       self.true_outputs: Y,
                                       self.seq_len : seq_len
                                   })                
            print(cost)
                
                
    def predict(self, x):
        
        m = x.shape[0]
        t_x = x.shape[1]
        seq_len = [t_x] * m
    

        return self.sess.run([self.y_pred], 
                             feed_dict={
                                 self.inputs : x,
                                 self.seq_len : seq_len
                             })
        