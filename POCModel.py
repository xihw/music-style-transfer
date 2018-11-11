
# coding: utf-8

# In[1]:


import pypianoroll as pr
import tensorflow as tf
import numpy as np
import os


# In[2]:


def load_data(folder):
    for filename in os.listdir(folder):
        print(os.path.join(folder, filename))
        return pr.parse(os.path.join(folder, filename))
        break
        


# In[11]:


midi = pr.parse('./data/jazz/Chelsea Bridge.mid')
#midi.trim_trailing_silence()
midi.tempo.shape


# In[12]:


midi.plot()


# In[13]:


dat = load_data('./data/jazz/')


# In[5]:


datY = dat.tracks[0].pianoroll / 128.


# In[7]:


datY.shape


# In[8]:


dat.binarize()
datX = dat.tracks[0].pianoroll


# In[10]:


datX.shape


# In[144]:


datX = datX.reshape(48, 100, 128)
datY = datY.reshape(48, 100, 128)


# In[161]:


class genModel():
    def __init__(self):
        self.input_size = 128
        self.hidden_unit = 256
        self.output_size = 128
        self.keep_prob = 0.8
    
    def simpleLSTM(self):
    
    
        # reset graph
        tf.reset_default_graph()
    
        # input output unit    
        self.inputs = tf.placeholder(tf.float32, (None, None, input_size))
        self.true_outputs = tf.placeholder(tf.float32, (None, None, output_size))
    
    
    
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
        self.W = tf.get_variable("W", shape=(self.input_size, self.hidden_unit), initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable("b", shape=(self.input_size, 1), initializer=tf.zeros_initializer())

        # reshape lstm out to [hidden_unit, None], prepare for matrix multiplication
        self.z = tf.transpose(tf.reshape(self.lstm_out, (-1, self.hidden_unit)))
        self.z = tf.matmul(self.W, self.z) + self.b

        # use sigmoid function to map between [0, 1] as the input velocity has been mapped to [0, 1]
        self.y_pred = tf.sigmoid(self.z)

        # reshape true_outputs to the same dimension
        self.y_true = tf.transpose(tf.reshape(self.true_outputs, (-1, self.output_size)))

        # loss
        self.cost = tf.reduce_sum(tf.square(self.y_true - self.y_pred))
        
    
    def train_batch(self, datX, datY, epochs=100):
        
        m = datX.shape[0]
        Tx = datX.shape[1]
        seq_len = [Tx] * m
        
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
        
            # Run the initialization
            sess.run(init)
        
            # Do the training loop
            for epoch in range(epochs):
                opt, cost = sess.run([self.optimizer, self.cost], 
                             feed_dict={
                                 self.inputs : datX,
                                 self.true_outputs: datY,
                                 self.seq_len : seq_len
                             })                
                print opt, cost
                
            y_pred = sess.run([self.y_pred], 
                     feed_dict={
                     self.inputs : datX,
                     self.seq_len : seq_len
                     })
        return y_pred
                
    def predict(self, datX):
        
        m = datX.shape[0]
        Tx = datX.shape[1]
        seq_len = [Tx] * m
    
        with tf.Session() as sess:
            return sess.run([self.y_pred], 
                         feed_dict={
                         self.inputs : datX,
                         self.seq_len : seq_len
                     })


        


# In[61]:


d = pr.parse('./data/classical/bach_846_format0.mid')
x = d.copy()
x.binarize()
y = d


# In[162]:


model = genModel()
#lstm_out, lstm_state, y, true_y, loss = genModel(1)
model.simpleLSTM()


# In[175]:


pred_y = model.train_batch(datX, datY, 1000)


# In[176]:


yy = np.reshape(np.transpose(pred_y[0]), (48, 100, 128))


# In[177]:


np.max(np.abs((datY-yy)*128))


# In[109]:


adam = tf.train.AdamOptimizer().minimize(loss)


# In[ ]:


with tf.Session() as sess:
    sess.run([adam], feed_dict={inputs: datX, ou})


# In[ ]:


tf.


# In[20]:


lstm_out_reshape = tf.reshape(lstm_out, (-1, hidden_unit))


# In[21]:


z = tf.matmul(W, lstm_out_reshape)


# In[22]:


z = z + b


# In[23]:


y = tf.sigmoid(z) * 127


# In[24]:


y

