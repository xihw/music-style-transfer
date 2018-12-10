from keras.models import Model
from keras.layers import Input, Conv1D, LSTM, Dropout, Dense, Lambda, multiply
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K


class Seq2SeqLSTM():
    def __init__(self, num_timestamps=1000):
        self.input_shape = (num_timestamps, 128)
        self.output_shape = (num_timestamps, 128)
        self.init_hyper_params()
        

    def init_hyper_params(self):
        self.encoder_conv_filters = 128
        self.encoder_lstm_unit = 128
        self.encoder_keep_prob = 0.8
        self.style_lstm_unit = 64
        self.style_keep_prob = 0.8
        self.learning_rate = 0.0015
        self.alpha_zero_loss = 10


    def encoder_conv_layer(self, inputs):
        outputs = Conv1D(filters=self.encoder_conv_filters, kernel_size=2, strides=1, padding="same", activation='relu', input_shape=inputs.shape)(inputs)
        return outputs


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


    def custom_loss(self, y_true, y_pred):
        zeros = K.zeros_like(y_true)
        zeroLoss = K.mean(multiply([
            K.cast(K.equal(y_true, zeros), dtype='float32'), 
            K.square(y_true - y_pred)]))
        oneLoss = K.mean(multiply([
            K.cast(K.not_equal(y_true, zeros), dtype='float32'), 
            K.square(y_true - y_pred)]))
        return zeroLoss * self.alpha_zero_loss + oneLoss
            

    def prepare(self):
        X = Input(shape=self.input_shape)

        Y_pred = X
        Y_pred = self.encoder_conv_layer(Y_pred)
        Y_pred = self.encoder_lstm_layer(Y_pred)
        Y_pred = self.style_lstm_layer(Y_pred)
        Y_pred = self.dense_output_layer(Y_pred)
        
        self.model = Model(inputs=X, outputs=Y_pred)
        adam_opt = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss=self.custom_loss, optimizer=adam_opt)
        self.model.summary()
    
    def train(self, X_train, Y_true, epochs=50, batch_size=32):
        history = self.model.fit(X_train, Y_true, epochs=epochs, batch_size=batch_size, shuffle=True)

        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.savefig('output/tmp.png')

        self.model.save_weights("output/Seq2SeqModel.h5")
        print("model saved to disk")
                
                
    def predict(self, x):
        y_pred = self.model.predict(x)
        y_pred = [np.round(yt) for yt in y_pred]
        return y_pred[0]
       
       
