import tensorflow as tf
from time import time
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer, Layer, MaxPool2D, AveragePooling2D,\
    BatchNormalization, Dropout, ReLU, LeakyReLU, Activation
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow_probability as tfp


INIT_M = 3.



class LotteryLayer(Layer):

    def __init__(self, kernel_init_constant=False, trainable_kernel=False, **kwargs):
        self.trainable_kernel = trainable_kernel
        self.kernel_init_constant = kernel_init_constant 

        super().__init__(**kwargs)

    def build(self, input_shape, shape):
        M_init = tf.constant_initializer(INIT_M)

        self.W = self.add_weight('W', shape=shape, trainable=self.trainable_kernel, initializer=tf.keras.initializers.GlorotNormal())
        self.W_rec = self.W.numpy()
        if self.kernel_init_constant:
            self.to_signed_constant()

        self.M = self.add_weight('M', shape=shape, trainable=True, initializer=M_init)
        super().build(input_shape)


    def to_signed_constant(self):
        # makes kernel weights either -self.std or +self.std
        new_w = np.full(self.W.shape, -self.std, dtype=np.float32)*(self.W_rec<0)+np.full(self.W.shape, self.std, dtype=np.float32)*(self.W_rec>=0)
        self.W.assign(new_w)


    def resample_masked(self):
        new_W_rec = tf.cast(tf.keras.initializers.GlorotNormal()(self.W.shape), dtype=tf.float32).numpy()
        new_new_w = np.full(self.W.shape, -self.std, dtype=np.float32)*(new_W_rec<0)+np.full(self.W.shape, self.std, dtype=np.float32)*(new_W_rec>=0)

        mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.float32)

        new_new_w = self.W*mask+new_new_w*(1-mask)

        self.W.assign(new_new_w)


    def reset_mask(self):
        M_init = tf.constant_initializer(INIT_M)
        self.M.assign(M_init(self.M.shape))

    def get_mask(self, training, inverse_mask=False, use_mask=True):
        if not use_mask:
            return None
        mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.float32)
        # mask = tf.sigmoid(self.M)
        

        if inverse_mask:
            mask = 1-mask #? Before or after training?
        
        if training:
            mask += tf.sigmoid(self.M) - tf.stop_gradient(tf.sigmoid(self.M)) # Trick to let gradients pass
        
        return mask


    def get_int_mask(self):
        # r = tf.random.uniform(shape=self.M.shape, minval=0, maxval=1)
        # m = tf.math.greater(tf.sigmoid(self.M), r)
        # m = tf.dtypes.cast(m, tf.int32)
        m = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.int32)
        return m

    def get_rescaling_factor(self, mask):
        tot = np.prod(self.M.shape).astype(np.float32)
        n_nonz = tf.math.count_nonzero(mask)
        return tf.cast(tot/n_nonz, dtype=tf.float32) # Dynamic weight rescaling


class LotteryConv2D(LotteryLayer):

    def __init__(self, filters, kernel_size, strides=1, padding='VALID', kernel_init_constant=False, trainable_kernel=False, **kwargs):
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        super().__init__(trainable_kernel=trainable_kernel, kernel_init_constant=kernel_init_constant, **kwargs)


    def build(self, input_shape):
        shape = (self.kernel_size, self.kernel_size, input_shape[-1], self.filters)
        self.std = np.sqrt(2/(np.prod(shape[:-1])+shape[-1]))
        super().build(input_shape, shape)


    def call(self, x, training=True, use_mask=True, inverse_mask=False, **kwargs):
        mask = self.get_mask(training, inverse_mask, use_mask)
        if use_mask:
            true_w = tf.math.multiply(mask, self.W)
            true_w *= self.get_rescaling_factor(mask)
        else:
            true_w = self.W

        out = tf.nn.conv2d(x, true_w, self.strides, self.padding)

        return out 



class LotteryDense(LotteryLayer):

    def __init__(self, units, kernel_init_constant=False, trainable_kernel=False, **kwargs):
        self.units = units
        super().__init__(kernel_init_constant=kernel_init_constant, trainable_kernel=trainable_kernel, **kwargs)

    def build(self, input_shape):
        shape = (input_shape[-1], self.units)
        self.std = np.sqrt(2/(sum(shape)))

        super().build(input_shape, shape)


    def call(self, x, training=True, use_mask=True, inverse_mask=False, **kwargs):
        mask = self.get_mask(training, inverse_mask, use_mask)
        if use_mask:
            true_w = tf.math.multiply(mask, self.W)
            true_w *= self.get_rescaling_factor(mask)
        else:
            true_w = self.W
        out = tf.keras.backend.dot(x, true_w)
        return out

