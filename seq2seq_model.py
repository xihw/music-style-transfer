from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, Lambda
from keras.optimizers import Adam

class Seq2SeqLSTM():
    def __init__(self, num_timestamps):
        self.input_shape = (num_timestamps, 128)
        self.output_shape = (num_timestamps, 128)
        self.init_hyper_params()
        

    def init_hyper_params(self):
        self.encoder_lstm_unit = 128
        self.encoder_keep_prob = 0.8
        self.style_lstm_unit = 64
        self.style_keep_prob = 0.8
        self.learning_rate = 0.0015


    def encoder_lstm_layer(self, inputs):
        outputs = LSTM(units=self.encoder_lstm_unit, return_sequences=True)(inputs)
        outputs = Dropout(self.encoder_keep_prob)(outputs)
        return outputs


    def style_lstm_layer(self, inputs):
        outputs = LSTM(units=self.style_lstm_unit, return_sequences=True)(inputs)
        outputs = Dropout(self.style_keep_prob)(outputs)
        return outputs


    def dense_output_layer(self, inputs):
        outputs = Dense(units=self.output_shape[1], activation='sigmoid')(inputs)
        # scale output from (0, 1) to (0, 127) in accordance with velocity range
        outputs = Lambda(lambda x: x * 127)(outputs)
        return outputs
            

    def prepare(self):
        X = Input(shape=self.input_shape)

        Y_pred = X
        Y_pred = self.encoder_lstm_layer(Y_pred)
        Y_pred = self.style_lstm_layer(Y_pred)
        Y_pred = self.dense_output_layer(Y_pred)
        
        self.model = Model(inputs=X, outputs=Y_pred)
        adam_opt = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='mean_squared_error', optimizer=adam_opt, metrics=['accuracy'])
        self.model.summary()
    
    def train(self, X_train, Y_true, epochs=50, batch_size=32):
        self.model.fit(X_train, Y_true, epochs=epochs, batch_size=batch_size, shuffle=True)
                
                
    def predict(self, x):
        return self.model.predict(x)
       
       
