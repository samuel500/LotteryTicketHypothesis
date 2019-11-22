import tensorflow as tf
from time import time
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer, Layer, MaxPool2D, AveragePooling2D,\
    BatchNormalization, Dropout, ReLU, LeakyReLU, Activation
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow_probability as tfp



class BinaryDense(Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        super().__init__(**kwargs)

    def build(self, input_shape):
        shape = (input_shape[-1], self.units)
        self.std = np.sqrt(2/(np.prod(shape[:-1])+shape[-1]))

        M_init = tf.constant_initializer(0)

        self.M = self.add_weight('M', shape=shape, trainable=True, initializer=M_init)
        super().build(input_shape)


    def call(self, x, training=True, **kwargs):
        mask = self.get_weight(training)

        out = tf.keras.backend.dot(x, mask)
        return out

    def get_weight(self, training):

        mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.float32)
        mask *= 2 
        mask -= 1 
        if training:
            mask += 2*tf.sigmoid(self.M)-1 - tf.stop_gradient(2*tf.sigmoid(self.M)-1) # Trick to let gradients pass
        
        mask *= self.std
        return mask



# class BinaryLottery(Layer):
    


class BinaryLotteryDense(Layer):

    def __init__(self, units, kernel_init_constant=False, trainable_kernel=False, **kwargs):
        self.units = units
        super().__init__(**kwargs)


    def build(self, input_shape):
        shape = (input_shape[-1], self.units)
        self.std = np.sqrt(2/(np.prod(shape[:-1])+shape[-1]))
        M_init = tf.constant_initializer(5)
        WM_init = tf.constant_initializer(0)

        self.M = self.add_weight('M', shape=shape, trainable=True, initializer=M_init)
        self.WM = self.add_weight('WM', shape=shape, trainable=True, initializer=WM_init)

        super().build(input_shape)


    def call(self, x, training=True, **kwargs):

        mask = self.get_mask(training)
        weight = self.get_weight(training)

        true_w = tf.math.multiply(mask, weight)
        true_w *= self.get_rescaling_factor(mask)

        out = tf.keras.backend.dot(x, true_w)
        return out


    def get_weight(self, training):

        weight = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.WM)).sample(), dtype=tf.float32)
        weight *= 2 
        weight -= 1

        if training:
            weight += 2*tf.sigmoid(self.WM)-1 - tf.stop_gradient(2*tf.sigmoid(self.WM)-1) # Trick to let gradients pass
        
        weight *= self.std
        return weight


    def get_mask(self, training, inverse_mask=False, use_mask=True):

        mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.float32)
        
        if training:
            mask += tf.sigmoid(self.M) - tf.stop_gradient(tf.sigmoid(self.M)) # Trick to let gradients pass
        
        return mask

    def get_rescaling_factor(self, mask):
        tot = np.prod(self.M.shape).astype(np.float32)
        n_nonz = tf.math.count_nonzero(mask)
        return tf.cast(tot/n_nonz, dtype=tf.float32) # Dynamic weight rescaling

    def get_int_mask(self):
        m = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.int32)
        return m



class BinaryLotteryConv2D(Layer):

    def __init__(self, filters, kernel_size, strides=1, padding='VALID',
                    trainable_WM=True, trainable_M=True, const_init_WM=0, const_init_M=5,
                    **kwargs):
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.trainable_WM = trainable_WM
        self.trainable_M = trainable_M

        self.const_init_WM = const_init_WM
        self.const_init_M = const_init_M

        super().__init__(**kwargs)


    def build(self, input_shape):
        shape = (self.kernel_size, self.kernel_size, input_shape[-1], self.filters)

        self.std = np.sqrt(2/(np.prod(shape[:-1])+shape[-1]))

        M_init = tf.constant_initializer(self.const_init_M)
        WM_init = tf.constant_initializer(self.const_init_WM)

        self.M = self.add_weight('M', shape=shape, trainable=self.trainable_M, initializer=M_init)
        self.WM = self.add_weight('WM', shape=shape, trainable=self.trainable_WM, initializer=WM_init)

        super().build(input_shape)


    def call(self, x, training=True, **kwargs):
        mask = self.get_mask(training)
        weight = self.get_weight(training)

        true_w = tf.math.multiply(mask, weight)
        true_w *= self.get_rescaling_factor(mask)

        out = tf.nn.conv2d(x, true_w, self.strides, self.padding)
        # out *= self.get_rescaling_factor(mask)
        return out 


    def get_weight(self, training, rescale=True):

        weight = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.WM)).sample(), dtype=tf.float32)
        weight *= 2 
        weight -= 1

        if training:
            weight += 2*tf.sigmoid(self.WM)-1 - tf.stop_gradient(2*tf.sigmoid(self.WM)-1) # Trick to let gradients pass
        if rescale:
            weight *= self.std
        return weight


    def get_mask(self, training, inverse_mask=False, use_mask=True):

        mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.float32)
        
        if training:
            mask += tf.sigmoid(self.M) - tf.stop_gradient(tf.sigmoid(self.M)) # Trick to let gradients pass
        
        return mask

    def get_rescaling_factor(self, mask):
        tot = np.prod(self.M.shape).astype(np.float32)
        n_nonz = tf.math.count_nonzero(mask)
        return tf.cast(tot/n_nonz, dtype=tf.float32) # Dynamic weight rescaling

    def get_int_mask(self):
        m = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.int32)
        return m

