import tensorflow as tf
from time import time
import numpy as np
from itertools import product

from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer, Layer, MaxPool2D, AveragePooling2D,\
    BatchNormalization, Dropout, ReLU, LeakyReLU, Activation
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow_probability as tfp


mnist = tf.keras.datasets.mnist
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, 'train samples')
print(x_test.shape[0], 'test samples')


x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]



train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)



loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')




class LotteryDense(Layer):

    def __init__(self, units, trainable_kernel=False, **kwargs):
        self.units = units
        self.trainable_kernel = trainable_kernel
        super(LotteryDense, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = (input_shape[-1], self.units)
        M_init = tf.constant_initializer(1.)


        self.W = self.add_weight('W', shape=shape, trainable=self.trainable_kernel, initializer=tf.keras.initializers.GlorotNormal())
        self.W_rec = self.W.numpy()
        self.std = np.sqrt(2/(sum(shape)))
      
        self.M = self.add_weight('M', shape=shape, trainable=True, initializer=M_init)

        super().build(input_shape)


    def get_mask(self):
        # r = tf.random.uniform(shape=self.M.shape, minval=0, maxval=1)
        # m = tf.math.greater(tf.sigmoid(self.M), r)
        # m = tf.dtypes.cast(m, tf.int32)
        m = tf.cast(tfp.distributions.Bernoulli(probs=tf.nn.sigmoid(self.M)).sample(), dtype=tf.int32)

        return m

    def to_signed_constant(self):
        new_w = np.full(self.W.shape, -self.std, dtype=np.float32)*(self.W_rec<0)+np.full(self.W.shape, self.std, dtype=np.float32)*(self.W_rec>=0)
        self.W.assign(new_w)


    def call(self, x, training=True, **kwargs):
        out = None
        if training:
            mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.float32)\
                + tf.nn.sigmoid(self.M)\
                - tf.stop_gradient(tf.nn.sigmoid(self.M)) # Trick to let gradients pass
            # m = tf.sigmoid(self.M)
            true_w = tf.math.multiply(mask, self.W)

            tot = np.prod(self.M.shape).astype(np.float32)
            n_nonz = tf.math.count_nonzero(mask)
            true_w *= tf.cast(tot/n_nonz, dtype=tf.float32) # Dynamic weight rescaling


            out = tf.keras.backend.dot(x, true_w)


        else:
            #r = tf.random.uniform(shape=self.M.shape, minval=0, maxval=1)
            #m = tf.math.greater(tf.sigmoid(self.M), r)
            #m = tf.dtypes.cast(m, tf.float32)
            mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.nn.sigmoid(self.M)).sample(), dtype=tf.float32)

            #print(tf.math.count_nonzero(m))
            
            #m = np.random.binomial(1, tf.sigmoid(self.M), self.M.shape)
            true_w = tf.math.multiply(mask, self.W)

            tot = np.prod(self.M.shape).astype(np.float32)
            n_nonz = tf.math.count_nonzero(mask)
            true_w *= tf.cast(tot/n_nonz, dtype=tf.float32)


            out = tf.keras.backend.dot(x, true_w)
        return out


class LotteryConv2D(Layer):

    def __init__(self, filters, kernel_size, strides=1, padding='VALID', trainable_kernel=False, **kwargs):
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.trainable_kernel = trainable_kernel

        super(LotteryConv2D, self).__init__(**kwargs)


    def build(self, input_shape):

        shape = (self.kernel_size, self.kernel_size, input_shape[-1], self.filters)

        M_init = tf.constant_initializer(1.)

        self.W = self.add_weight('W', shape=shape, trainable=self.trainable_kernel, initializer=tf.keras.initializers.GlorotNormal())
        self.W_rec = self.W.numpy()
        self.std = np.sqrt(2/(np.prod(shape[:-1])+shape[-1]))

        self.M = self.add_weight('M', shape=shape, trainable=True, initializer=M_init)
        super().build(input_shape)


    def call(self, x, training=True, **kwargs):
        if training:
            mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.sigmoid(self.M)).sample(), dtype=tf.float32)\
                + tf.nn.sigmoid(self.M)\
                - tf.stop_gradient(tf.nn.sigmoid(self.M)) # Trick to let gradients pass
            # m = tf.sigmoid(self.M)
            true_w = tf.math.multiply(mask, self.W)

            tot = np.prod(self.M.shape).astype(np.float32)
            n_nonz = tf.math.count_nonzero(mask)
            true_w *= tf.cast(tot/n_nonz, dtype=tf.float32) # Dynamic weight rescaling


            out = tf.nn.conv2d(
                x,
                true_w,
                self.strides,
                self.padding
            )

        else:
            #r = tf.random.uniform(shape=self.M.shape, minval=0, maxval=1)
            #m = tf.math.greater(tf.sigmoid(self.M), r)
            #m = tf.dtypes.cast(m, tf.float32)
            mask = tf.cast(tfp.distributions.Bernoulli(probs=tf.nn.sigmoid(self.M)).sample(), dtype=tf.float32)

            #print(tf.math.count_nonzero(m))
            
            #m = np.random.binomial(1, tf.sigmoid(self.M), self.M.shape)
            true_w = tf.math.multiply(mask, self.W)

            tot = np.prod(self.M.shape).astype(np.float32)
            n_nonz = tf.math.count_nonzero(mask)
            true_w *= tf.cast(tot/n_nonz, dtype=tf.float32)

            out = tf.nn.conv2d(
                x,
                true_w,
                self.strides,
                self.padding
            )

        return out 

    def to_signed_constant(self):
        new_w = np.full(self.W.shape, -self.std, dtype=np.float32)*(self.W_rec<0)+np.full(self.W.shape, self.std, dtype=np.float32)*(self.W_rec>=0)
        self.W.assign(new_w)

    def get_mask(self):
        # r = tf.random.uniform(shape=self.M.shape, minval=0, maxval=1)
        # m = tf.math.greater(tf.sigmoid(self.M), r)
        # m = tf.dtypes.cast(m, tf.int32)
        m = tf.cast(tfp.distributions.Bernoulli(probs=tf.nn.sigmoid(self.M)).sample(), dtype=tf.int32)
        return m



#@tf.function
def train_step(images, labels, optimizer, trainable_variables):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        #kl_loss = -0.5*tf.math.reduce_sum((1 + logvar - tf.square(mu) - tf.exp(logvar)))

        #kl_loss = 0.00012*tf.reduce_mean(kl_loss)

        loss = loss_object(labels, predictions)
        #loss += kl_loss
        #print(loss)
        #print(kl_loss)
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)



def get_all_masks(layers):
    masks = []
    for layer in layers:
        if type(layer) in {LotteryDense, LotteryConv2D}:
            masks.append(layer.M)
    return masks

def get_all_kernels(layers):
    kernels = []
    for layer in layers:
        if type(layer) in {LotteryConv2D, LotteryDense}:
            kernels.append(layer.W)
    return kernels

def print_p_pruned(layers):

    tot_w = 0
    tot_m = 0
    for i, l in enumerate(layers):
        if type(l) is LotteryDense or type(l) is LotteryConv2D:
            tot = np.prod(l.M.shape)
            tot_w += tot
            mask = l.get_mask()
            #print(mask)
            m = tf.math.count_nonzero(mask).numpy()
            tot_m += m
            print('Layer', i, 'p pruned:', 1-m/tot)


    print('Tot p pruned:', 1-tot_m/tot_w)


layers = [
    InputLayer(input_shape=(28, 28, 1)),
    LotteryConv2D(16, 3, strides=2),
    LeakyReLU(),

    LotteryConv2D(32, 3, strides=2),
    LeakyReLU(),

    LotteryConv2D(64, 3, strides=1),
    LeakyReLU(),

    LotteryConv2D(128, 3, strides=2),
    LeakyReLU(),

    Flatten(),
    #Dense(300, activation='relu', trainable=False),
    #Dense(100, activation='relu', trainable=False),
    # LotteryDense(256),
    # LeakyReLU(),
    # Dropout(0.3),
    LotteryDense(10),
    Activation('softmax')
]


model = tf.keras.Sequential(
    layers
)

model.summary()




if __name__=='__main__':
    kernel_optimizer = tf.keras.optimizers.Adam()
    mask_optimizer = tf.keras.optimizers.SGD(100, momentum=0.9)


    EPOCHS = 200

    for epoch in range(EPOCHS):
        
        # get_all_masks(model.layers)
        # raise
        if epoch==0:
            print('ttt')
            for i, l in enumerate(model.layers):
                if type(l) is LotteryDense or type(l) is LotteryConv2D:
                    l.to_signed_constant()



        st = time()
        for images, labels in train_ds:
            train_step(images, labels, get_all_masks(model.layers))

        print_p_pruned(model.layers)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)



        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1,
                            round(float(train_loss.result()), 4),
                            round(float(train_accuracy.result()*100), 3),
                            round(float(test_loss.result()), 4),
                            round(float(test_accuracy.result()*100), 2)))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        print('T:', time()-st)

