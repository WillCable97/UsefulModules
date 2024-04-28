import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding, LSTM, Dense, BatchNormalization

def RNN_model(vocab_size:int, embedding_dim: int, rnn_units:int, batch_size:int) -> tf.keras.models.Model:
    stateful_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None], mask_zero=True), 
        tf.keras.layers.SimpleRNN(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return stateful_model

def GRU_model(vocab_size:int, embedding_dim: int, rnn_units:int, batch_size:int) -> tf.keras.models.Model:
    stateful_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim), 
        tf.keras.layers.GRU(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return stateful_model


class LSTM_model(Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size, dropout_rate = 0.1):
        super().__init__()
        self.emb_layer = Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None])
        self.lstm_layer = LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform', dropout=dropout_rate)
        #self.batch_norm = BatchNormalization()
        self.dense_comp = Dense(vocab_size)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, input):
        input = self.emb_layer(input)
        input = self.lstm_layer(input)
        #input = self.batch_norm(input)
        input = self.dense_comp(input)
        return input
    