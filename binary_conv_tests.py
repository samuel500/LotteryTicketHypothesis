import tensorflow as tf
from time import time
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer, Layer, MaxPool2D, AveragePooling2D,\
    BatchNormalization, Dropout, ReLU, LeakyReLU, Activation
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow_probability as tfp


from binary_layers import *
from lottery_layers import *
from maxout import *
from trainable_dropout import *

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

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)


cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



#@tf.function
def train_step(images, labels, optimizer, trainable_variables):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)

        loss = cross_entropy(labels, predictions)

    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)



#@tf.function
def test_step(images, labels, use_mask=True):
    predictions = model(images, training=False, use_mask=use_mask)
    t_loss = cross_entropy(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)




layers = [
    InputLayer(input_shape=(28, 28, 1)),

    BinaryLotteryConv2D(512, kernel_size=3, strides=2),
    TrainableChannelDropout(),
    ReLU(),

    Flatten(),
    Dense(32),
    ReLU(),
    Dense(10),

    Activation('softmax')
]


model = LotteryModel(
    layers
)

model.summary()



if __name__=='__main__':


    kernel_optimizer = tf.keras.optimizers.Adam(3e-4)
    mask_optimizer = tf.keras.optimizers.SGD(100, momentum=0.9)


    EPOCHS = 1000


    all_masks = [] 

    for epoch in range(EPOCHS):

        st = time()

        for i, (images, labels) in enumerate(tqdm(train_ds)):

            train_step(images, labels, mask_optimizer, model.trainable_variables)


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

