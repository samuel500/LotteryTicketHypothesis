import tensorflow as tf
from time import time
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer, Layer, MaxPool2D, AveragePooling2D,\
    BatchNormalization, Dropout, ReLU, LeakyReLU, Activation
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow_probability as tfp

from layers import *


#tf.random.set_seed(8)


mnist = tf.keras.datasets.mnist
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, 'train samples')
print(x_test.shape[0], 'test samples')


x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

n_train = 30000
#print('n:', n_train)
x_train1 = x_train #[:n_train]
y_train1 = y_train #[:n_train]
x_train2 = x_train #[n_train:]
y_train2 = y_train #[n_train:]


train_ds1 = tf.data.Dataset.from_tensor_slices((x_train1, y_train1)).shuffle(10000).batch(128)

train_ds2 = tf.data.Dataset.from_tensor_slices((x_train2, y_train2)).shuffle(10000).batch(128)


test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1024)



loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')




#@tf.function
def train_step(images, labels, optimizer, trainable_variables, reg=0, use_mask=True, inverse_mask=False):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True, use_mask=use_mask, inverse_mask=inverse_mask)

        loss = loss_object(labels, predictions)

        if reg:
            reg_loss = 0
            for layer in model.layers:
                if type(layer) in {LotteryDense, LotteryConv2D}:
                    reg_loss += tf.reduce_sum(layer.M)
            reg_loss *= reg
            loss += reg_loss

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
        if type(layer) in {LotteryDense, LotteryConv2D, TrainableDropout, TrainableChannelDropout}:
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
        if type(l) in {LotteryDense, LotteryConv2D, TrainableDropout, TrainableChannelDropout}:
            tot = np.prod(l.M.shape)
            tot_w += tot
            mask = l.get_mask()
            #print(mask)
            m = tf.math.count_nonzero(mask).numpy()
            tot_m += m
            print('Layer', i, 'p pruned:', 1-m/tot)


    print('Tot p pruned:', 1-tot_m/tot_w)


lott_t = True

layers = [
    InputLayer(input_shape=(28, 28, 1)),

    # LotteryConv2D(32, 3, strides=2, trainable_kernel=lott_t),
    Conv2D(16, 3, strides=2),
    LeakyReLU(),
    TrainableChannelDropout(),


    # LotteryConv2D(64, 3, strides=2, trainable_kernel=lott_t),
    Conv2D(32, 3, strides=2),
    LeakyReLU(),
    TrainableChannelDropout(),


    # LotteryConv2D(128, 3, strides=2, trainable_kernel=lott_t),
    Conv2D(64, 3, strides=2),
    LeakyReLU(),
    TrainableChannelDropout(),


    # LotteryConv2D(256, 3, strides=2, trainable_kernel=lott_t),
    # LeakyReLU(),
    # TrainableChannelDropout(),

    Flatten(),
    # Dense(300, activation='relu', trainable=False),
    # LeakyReLU(),
    # Dense(100, activation='relu', trainable=False),
    # LotteryDense(300, trainable_kernel=lott_t),
    # LeakyReLU(),    
    # LotteryDense(100, trainable_kernel=lott_t),
    # LeakyReLU(),    

    # # Dropout(0.3),
    # LotteryDense(10, trainable_kernel=lott_t),
    # TrainableDropout(),

    # Dense(300),
    # LeakyReLU(),
    # TrainableDropout(),
    # Dense(100),
    # LeakyReLU(),
    # TrainableDropout(),

    Dense(10),

    Activation('softmax')
]


model = LotteryModel(
    layers
)

model.summary()



if __name__=='__main__':
    kernel_optimizer = tf.keras.optimizers.Adam()
    mask_optimizer = tf.keras.optimizers.SGD(100, momentum=0.9)

    switch = 20

    EPOCHS = 500

    for epoch in range(EPOCHS):

        # if epoch==0:
        #     print('ttt')
        #     for i, l in enumerate(model.layers):
        #         if type(l) is LotteryDense or type(l) is LotteryConv2D:
        #             l.to_signed_constant()

        st = time()

        # if epoch < switch:


        for i, (images1, labels1) in enumerate(tqdm(train_ds1)):
            #train_step(images1, labels1, kernel_optimizer, get_all_kernels(model.layers), use_mask=False) # DropConnect! http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf
            train_step(images1, labels1, mask_optimizer, get_all_masks(model.layers), reg=5e-7)
            # if epoch<3:
            #     train_step(images1, labels1, mask_optimizer, model.trainable_variables, reg=True)
            # else:
            #     train_step(images1, labels1, mask_optimizer, model.trainable_variables, reg=True, )



        # for i, (images2, labels2) in enumerate(tqdm(train_ds2)):
        #     train_step(images2, labels2, mask_optimizer, get_all_masks(model.layers), reg=True)

        # else:
        #     for i, (images2, labels2) in enumerate(tqdm(train_ds2)):

        #         train_step(images2, labels2, mask_optimizer, get_all_masks(model.layers), reg=True)

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

