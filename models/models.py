from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GRU, CuDNNGRU, LSTM, CuDNNLSTM

from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model

import sys

########################################
## set directories and parameters
########################################

from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim


def get_rnn_model( rnn_type,
                   embedding_matrix, embed_dim,
                   maxlen, nb_words,
                   num_rnn_layers, rnn_units, rnn_dropout,
                   dense_units, dense_dropout,
                   bidirectional=False, attention=False,
                   cudnn=True,
                   summary=False):

    input_text = Input(shape=(maxlen,), dtype='int32')

    x = Embedding(nb_words, embed_dim,
                  weights=[embedding_matrix],
                  input_length=maxlen,
                  trainable=False)(input_text)

    # assert (rnn_layer=='GRU' or rnn_layer='LSTM')

    # Define rnn layer
    if cudnn:
        if rnn_type=='GRU':
            rnn_layer = CuDNNGRU(rnn_units, return_sequences=True)
        elif rnn_type=='LSTM':
            rnn_layer = CuDNNLSTM(rnn_units, return_sequences=True)
        else:
            print ( "Invalid entry for rnn_type. Must be 'LSTM' or 'GRU'")
    else:
        if rnn_type=='GRU':
            rnn_layer = GRU(rnn_units,
                            dropout=rnn_dropout, recurrent_dropout=rnn_dropout,
                            return_sequences=True)
        elif rnn_type=='LSTM':
            rnn_layer = LSTM(rnn_units,
                             dropout=rnn_dropout, recurrent_dropout=rnn_dropout,
                             return_sequences=True)
    if bidirectional:
        rnn_layer = Bidirectional(rnn_layer)

    for _ in range(num_rnn_layers):
        x = rnn_layer(x)

    if attention:
        x = Attention(maxlen)(x)
        x = Dropout(dense_dropout)(x)
    else:
        x = GlobalMaxPool1D()(x)
        x = Dropout(dense_dropout)(x)

    x = Dense(dense_units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dense_dropout)(x)
    preds = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=[input_text], outputs=preds)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    if summary: model.summary
    return model

def get_lstm_heir_model(embedding_matrix, embed_dim,
                        max_sent_len, max_sents, nb_words,
                        lstm_units): #, lstm_dropout, dense_dropout, dense_units):

    sentence_input = Input(shape=(max_sent_len,), dtype='int32')

    embedded_sequences = Embedding(nb_words,
                                   embed_dim,
                                   weights=[embedding_matrix],
                                   input_length=max_sent_len,
                                   trainable=False)(sentence_input)

    l_lstm = Bidirectional(LSTM(lstm_units))(embedded_sequences)

    sentEncoder = Model(sentence_input, l_lstm)

    review_input = Input(shape=(max_sents,max_sent_len), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)

    l_lstm_sent = Bidirectional(LSTM(lstm_units))(review_encoder)

    preds = Dense(6, activation='sigmoid')(l_lstm_sent)

    model = Model(review_input, preds)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def get_dpcnn_model(embedding_matrix, embedding_size,
                    maxlen, nb_words,
                    filter_nr, kernel_size, repeat_block, dropout_convo,
                    dense_size, repeat_dense, dropout_dense,
                    # l2_reg_convo, l2_reg_dense, use_prelu,
                    trainable_embedding, use_batch_norm):
    """
    Note:
        Implementation of http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
        post activation is used instead of pre-activation, could be worth exploring
    """

    input_text = Input(shape=(maxlen,))

    if embedding_matrix is not None:
        embedding = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=trainable_embedding)(
            input_text)
    else:
        embedding = Embedding(nb_words, embedding_size)(input_text)

    x = _convolutional_block(filter_nr, kernel_size)(embedding) #, use_batch_norm, use_prelu, dropout_convo, l2_reg_convo)(embedding)
    x = _convolutional_block(filter_nr, kernel_size)(x) #, use_batch_norm, use_prelu, dropout_convo, l2_reg_convo)(x)

    if embedding_size == filter_nr:
        x = add([embedding, x])
    else:
        embedding_resized = _shape_matching_layer(filter_nr, dropout_convo)(embedding) #, use_prelu, dropout_convo)(embedding) #, l2_reg_convo)(embedding)
        x = add([embedding_resized, x])

    for _ in range(repeat_block):
        x = _dpcnn_block(filter_nr, kernel_size, dropout_convo)(x) # , use_batch_norm, dropout_convo) #, use_prelu, dropout_convo, l2_reg_convo)(x)

    x = GlobalMaxPool1D()(x)
    for _ in range(repeat_dense):
        x = _dense_block(dense_size, use_batch_norm, dropout_dense)(x) #, use_prelu, dropout_dense, l2_reg_dense)(x)
    predictions = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_text, outputs=predictions)

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

def _dense_block(dense_size, use_batch_norm, dropout): #, use_prelu, dropout, l2_reg):
    def f(x):
#         x = Dense(dense_size, activation='linear',
#                   kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = Dense(dense_size, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
#         x = _bn_relu_dropout_block(use_batch_norm, use_prelu, dropout)(x)
        return x

    return f

def _convolutional_block(filter_nr, kernel_size): #, use_batch_norm, use_prelu, dropout, l2_reg):
    def f(x):
#         x = Conv1D(filter_nr, kernel_size=kernel_size, padding='same', activation='relu',
#                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
#                    kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = Conv1D(filter_nr, kernel_size=kernel_size, padding='same',
                   activation='relu')(x)
#         x = _bn_relu_dropout_block(use_batch_norm, use_prelu, dropout)(x)
        x = BatchNormalization()(x)
        return x

    return f

def _dpcnn_block(filter_nr, kernel_size, dropout): #, use_batch_norm, use_prelu, dropout, l2_reg):
    def f(x):
        x = MaxPooling1D(pool_size=3, stride=2)(x)
        main = _convolutional_block(filter_nr, kernel_size)(x) #, use_batch_norm, use_prelu, dropout, l2_reg)(x)
        main = _convolutional_block(filter_nr, kernel_size)(main) #, use_batch_norm, use_prelu, dropout, l2_reg)(main)
        x = add([main, x])
        return x

    return f

def _shape_matching_layer(filter_nr, dropout): # use_prelu, dropout, l2_reg):
    def f(x):
        x = Conv1D(filter_nr, kernel_size=1, padding='same', activation='relu')(x)
#                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
#                    kernel_regularizer=regularizers.l2(l2_reg))(x)
#         if use_prelu:
#             x = PReLU()(x)
#         else:
#             x = Lambda(relu)(x)
        x = Dropout(dropout)(x)
        return x

    return f
