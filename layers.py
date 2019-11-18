import tensorflow as tf
from time import time
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer, Layer, MaxPool2D, AveragePooling2D,\
    BatchNormalization, Dropout, ReLU, LeakyReLU, Activation
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow_probability as tfp


INIT_M = 2.


class LotteryModel(Model):

    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs)
        #self.inputs = layers[0]
        self.seq_model = tf.keras.Sequential(layers)
        self.layers_list = layers

    def call(self, x, **kwargs):
        for layer in self.layers_list:
            if type(layer) in {LotteryDense, LotteryConv2D, TrainableDropout}: # careful
                x = layer(x, **kwargs)
            else:
                x = layer(x)
        return x

    def summary(self):
        self.seq_model.summary()



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


class BinaryLotteryDense(Layer):

    def __init__(self, units, kernel_init_constant=False, trainable_kernel=False, **kwargs):
        self.units = units
        super().__init__(**kwargs)


    def build(self, input_shape):
        shape = (input_shape[-1], self.units)
        self.std = np.sqrt(2/(np.prod(shape[:-1])+shape[-1]))
        M_init = tf.constant_initializer(5)
        WM_init = tf.constant_initializer(0)


        self.WM = self.add_weight('WM', shape=shape, trainable=True, initializer=WM_init)
        self.M = self.add_weight('M', shape=shape, trainable=True, initializer=M_init)

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

