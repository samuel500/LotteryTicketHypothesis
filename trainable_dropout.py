
import tensorflow as tf
from time import time
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer, Layer, MaxPool2D, AveragePooling2D,\
    BatchNormalization, Dropout, ReLU, LeakyReLU, Activation
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow_probability as tfp



class TrainableDropout(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        shape = input_shape[1:]
        M_init = tf.constant_initializer(7.)
        self.M = self.add_weight('M', shape=shape, trainable=True, initializer=M_init)


    def call(self, x, training=True, **kwargs):
        out = None
        if training:
            mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.float32)\
                + tf.sigmoid(self.M)\
                - tf.stop_gradient(tf.sigmoid(self.M)) # Trick to let gradients pass
        else:
            mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.float32)

        tot = np.prod(self.M.shape).astype(np.float32)
        n_nonz = tf.math.count_nonzero(mask)

        out = tf.math.multiply(x, mask)

        out *= tf.cast(tot/n_nonz, dtype=tf.float32)

        return out

    def get_int_mask(self):
        m = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.int32)

        return m

class TrainableChannelDropout(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        shape = input_shape[-1]
        M_init = tf.constant_initializer(7.)
        self.M = self.add_weight('M', shape=shape, trainable=True, initializer=M_init)


    def call(self, x, training=True, **kwargs):
        out = None
        if training:
            mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.float32)\
                + tf.sigmoid(self.M)\
                - tf.stop_gradient(tf.sigmoid(self.M)) # Trick to let gradients pass
        else:
            mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.float32)

        tot = np.prod(self.M.shape).astype(np.float32)
        n_nonz = tf.math.count_nonzero(mask)


        out = tf.einsum('bxyc,c->bxyc', x, mask)

        out *= tf.cast(tot/n_nonz, dtype=tf.float32)

        return out

    def get_int_mask(self):
        m = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.int32)

        return m



