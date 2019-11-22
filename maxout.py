import tensorflow as tf
from time import time
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer, Layer, MaxPool2D, AveragePooling2D,\
    BatchNormalization, Dropout, ReLU, LeakyReLU, Activation
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow_probability as tfp





class Maxout(Layer):

    def __init__(self, trainable_list=None, leaky_init=True, **kwargs):
        self.trainable_list = trainable_list if trainable_list is not None else [True, True, True, True] # W1, B1, W2, B2
        self.leaky_init = leaky_init
        super(Maxout, self).__init__(**kwargs)

    def build(self, input_shape):
        #print(input_shape)
        W1_t, B1_t, W2_t, B2_t = self.trainable_list
        W2_init = 'zeros'
        if self.leaky_init:
            W2_init = tf.constant_initializer(0.2)

        self.W1 = self.add_weight("W1", shape=(1, *input_shape[1:]), trainable=W1_t, initializer='ones')
        self.B1 = self.add_weight("B1", shape=(1, *input_shape[1:]), trainable=B1_t, initializer='zeros')
        self.W2 = self.add_weight("W2", shape=(1, *input_shape[1:]), trainable=W2_t, initializer=W2_init)
        self.B2 = self.add_weight("B2", shape=(1, *input_shape[1:]), trainable=B2_t, initializer='zeros')
        #print(self.W1)
        super().build(input_shape)


    def call(self, x, **kwargs):
        return tf.math.maximum(tf.math.multiply(x, self.W2)+self.B2, tf.math.multiply(x, self.W1)+self.B1)

    def compute_output_shape(self, input_shape):
        return input_shape


class ChannelMaxout(Layer):
    def __init__(self, trainable_list=None, leaky_init=True, **kwargs):
        self.trainable_list = trainable_list if trainable_list is not None else [True, True, True, True] # W1, B1, W2, B2
        self.leaky_init = leaky_init
        super(ChannelMaxout, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        W1_t, B1_t, W2_t, B2_t = self.trainable_list
        W2_init = 'zeros'
        if self.leaky_init:
            W2_init = tf.constant_initializer(0.2)

        self.W1 = self.add_weight("W1", shape=(input_shape[-1]), trainable=W1_t, initializer='ones')
        self.B1 = self.add_weight("B1", shape=(1, input_shape[-1]), trainable=B1_t, initializer='zeros')
        self.W2 = self.add_weight("W2", shape=(input_shape[-1]), trainable=W2_t, initializer=W2_init)
        self.B2 = self.add_weight("B2", shape=(1, input_shape[-1]), trainable=B2_t, initializer='zeros')
        #print(self.W1)
        super().build(input_shape)


    def call(self, x, **kwargs):
        #print('x.s', x.shape)
        #print('W1.s', self.W1.shape)
        return tf.math.maximum(tf.einsum('bxyc,c->bxyc', x, self.W2)+self.B2, tf.einsum('bxyc,c->bxyc', x, self.W1)+self.B1)

    def compute_output_shape(self, input_shape):
        return input_shape

