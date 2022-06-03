"""
# -*- coding: utf-8 -*-

Created on Jun 2022
@author: Prateek Yadav

"""
import tensorflow as tf
import tf2crf


class nerModel:

    def __init__(self, configObj):
        self.word_count = configObj['word_count']
        self.dense_embedding = configObj['dense_embedding']
        self.dense_units = configObj['dense_units']
        self.lstm_units = configObj['lstm_unit']
        self.lstm_dropout = configObj['lstm_dropout']
        self.batch_size = configObj['batch_size']
        self.max_epochs = configObj['max_epochs']
        self.tag_count = configObj['tag_count']
        self.max_len = configObj['max_len']

    def model(self):
        with tf.device('gpu'):
            input_layer = tf.keras.layers.Input(shape=(self.max_len,))
            model = tf.keras.layers.Embedding(self.word_count, self.dense_embedding,
                                              embeddings_initializer='uniform',
                                              input_length=self.max_len)(input_layer)
            model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units,
                                                                       recurrent_dropout=self.lstm_dropout,
                                                                       return_sequences=True))(model)
            model = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.dense_units,
                                      activation='relu'))(model)

            crf_layer = tf2crf.CRF(units=self.tag_count)
            output_layer = crf_layer(model)

            ner_model = tf.keras.models.Model(input_layer, output_layer)

            model_final = tf2crf.ModelWithCRFLoss(ner_model, sparse_target=True)

            opt = tf.keras.optimizers.Adam(learning_rate=0.001)

            model_final.compile(optimizer=opt)

            return model_final


if __name__ == '__main__':
    pass
