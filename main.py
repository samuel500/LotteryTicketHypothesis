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

n_train = 55000
print('n:', n_train)
x_train1 = x_train[:n_train]
y_train1 = y_train[:n_train]
x_train2 = x_train[n_train:]
y_train2 = y_train[n_train:]


train_ds1 = tf.data.Dataset.from_tensor_slices((x_train1, y_train1)).shuffle(10000).batch(128)
train_ds2 = tf.data.Dataset.from_tensor_slices((x_train2, y_train2)).shuffle(10000).batch(128)


test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1024)



cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')




#@tf.function
def train_step(images, labels, optimizer, trainable_variables, reg=0, use_mask=True, inverse_mask=False):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True, use_mask=use_mask, inverse_mask=inverse_mask)

        loss = cross_entropy(labels, predictions)

        if reg:
            reg_loss = 0
            for layer in model.layers:
                if type(layer) in {LotteryDense, LotteryConv2D, TrainableDropout, TrainableChannelDropout}:
                    reg_loss += tf.reduce_sum(layer.M)
            reg_loss *= reg
            loss += reg_loss

    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


#@tf.function
def train_steps(images, labels, weight_optimizer, mask_optimizer, reg=0, use_mask=True, inverse_mask=False):
    with tf.GradientTape() as weight_tape, tf.GradientTape() as mask_tape:
        predictions = model(images, training=True, use_mask=use_mask, inverse_mask=inverse_mask)

        prediction_loss = cross_entropy(labels, predictions)

        if reg:
            reg_loss = 0
            for layer in model.layers:
                if type(layer) in {LotteryDense, LotteryConv2D, TrainableDropout, TrainableChannelDropout}:
                    reg_loss += tf.reduce_sum(layer.M) #???? U sure?
            reg_loss *= reg
        else:
            reg_loss = 0

        mask_loss = prediction_loss + reg_loss
        
    trainable_weights = get_all_kernels(model.layers)+get_all_normals(model.layers)
    weight_gradients = weight_tape.gradient(prediction_loss, trainable_weights)
    weight_optimizer.apply_gradients(zip(weight_gradients, trainable_weights))

    trainable_masks = get_all_masks(model.layers)
    mask_gradients = mask_tape.gradient(mask_loss, trainable_masks)
    mask_optimizer.apply_gradients(zip(mask_gradients, trainable_masks))

    train_loss(mask_loss)
    train_accuracy(labels, predictions)



@tf.function
def test_step(images, labels, use_mask=True):
    predictions = model(images, training=False, use_mask=use_mask)
    t_loss = cross_entropy(labels, predictions)

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


def get_some_masks(layers, types):
    masks = []
    for layer in layers:
        if type(layer) in types:
            masks.append(layer.M)
    return masks


def get_all_normals(layers):
    normals = []
    for layer in layers:
        if type(layer) not in {LotteryConv2D, LotteryDense, TrainableDropout, TrainableChannelDropout}:
            if layer.trainable_variables:
                normals += layer.trainable_variables
    return normals


def print_p_pruned(layers):

    tot_w = 0
    tot_m = 0
    for i, l in enumerate(layers):
        if type(l) in {LotteryDense, LotteryConv2D, TrainableDropout, TrainableChannelDropout}:
            tot = np.prod(l.M.shape)
            tot_w += tot
            mask = l.get_int_mask()
            #print(mask)
            m = tf.math.count_nonzero(mask).numpy()
            tot_m += m
            print('Layer', i, '('+str(type(l))+')', 'p pruned:', 1-m/tot)


    print('Tot p pruned:', 1-tot_m/tot_w)


lott_t = True
kinic = False

layers = [
    InputLayer(input_shape=(28, 28, 1)),

    # #TrainableDropout(),
    LotteryConv2D(64, 3, strides=2, kernel_init_constant=kinic, trainable_kernel=lott_t),
    #Conv2D(16, 3, strides=2),
    LeakyReLU(),
    # TrainableDropout(),

    #TrainableChannelDropout(),


    LotteryConv2D(128, 3, strides=2, kernel_init_constant=kinic, trainable_kernel=lott_t),
    #Conv2D(32, 3, strides=2),
    LeakyReLU(),
    # TrainableDropout(),

    #TrainableChannelDropout(),


    LotteryConv2D(256, 3, strides=1, kernel_init_constant=kinic, trainable_kernel=lott_t),
    # Conv2D(64, 3, strides=2),
    LeakyReLU(),
    # TrainableDropout(),
    #TrainableChannelDropout(),


    LotteryConv2D(256, 3, strides=1, trainable_kernel=lott_t),
    LeakyReLU(),
    # # TrainableChannelDropout(),

    Flatten(),
    #TrainableDropout(),

    # Dense(32),
    # LotteryDense(1024, kernel_init_constant=kinic, trainable_kernel=lott_t),
    # LeakyReLU(),

    # LotteryDense(512, kernel_init_constant=kinic, trainable_kernel=lott_t),
    # LeakyReLU(),
    #TrainableDropout(),
    # Dense(100, activation='relu', trainable=False),
    # LotteryDense(300, trainable_kernel=lott_t),
    # LeakyReLU(),    
    # LotteryDense(100, trainable_kernel=lott_t),
    # LeakyReLU(),    

    # # Dropout(0.3),
    LotteryDense(64, kernel_init_constant=kinic, trainable_kernel=lott_t),
    LeakyReLU(),    

    # TrainableDropout(),

    # Dense(300),
    # LeakyReLU(),
    # TrainableDropout(),
    # Dense(100),
    # LeakyReLU(),
    # TrainableDropout(),
    #Dense(10),
    LotteryDense(10, kernel_init_constant=kinic, trainable_kernel=lott_t),

    Activation('softmax')
]


model = LotteryModel(
    layers
)

model.summary()



if __name__=='__main__':
    kernel_optimizer = tf.keras.optimizers.Adam()
    mask_optimizer = tf.keras.optimizers.SGD(100, momentum=0.9)

    switch = 40

    EPOCHS = 500

    test_use_mask = True

    for epoch in range(EPOCHS):

        # if epoch==0:
        #     print('ttt')
        #     for i, l in enumerate(model.layers):
        #         if type(l) is LotteryDense or type(l) is LotteryConv2D:
        #             l.to_signed_constant()

        st = time()

        if epoch < switch:
            for i, (images1, labels1) in enumerate(tqdm(train_ds1)):
                train_step(images1, labels1, kernel_optimizer, get_all_kernels(model.layers), use_mask=False) # DropConnect! http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf
                #train_step(images, labels, kernel_optimizer, model.trainable_variables)

        else:
            if epoch == switch:
                print("Switch!")  
            for i, (images2, labels2) in enumerate(tqdm(train_ds1)):
                train_step(images2, labels2, mask_optimizer, get_all_masks(model.layers))
        # if epoch:
        #     for i, l in enumerate(model.layers):
        #         if type(l) in {LotteryDense, LotteryConv2D}:

        # if not (epoch+1)%5:
        #     for i, l in enumerate(model.layers):
        #         if type(l) in {LotteryDense, LotteryConv2D}:
        #             l.resample_masked()
        #             mask = l.M.numpy()
        #             mask += 1.
        #             l.M.assign(mask)
        #             #l.reset_mask()

        # # for i, (images, labels) in enumerate(tqdm(train_ds)):

        #     #masks_to_train = get_some_masks(model.layers, {TrainableDropout})
        #     masks_to_train = get_all_masks(model.layers)
        #     #train_step(images, labels, mask_optimizer, masks_to_train, reg=1e-7)

        #     #train_steps(images, labels, kernel_optimizer, mask_optimizer, reg=1e-7, inverse_mask=True)

        #     # if epoch<3:
        #     train_step(images, labels, mask_optimizer, get_all_masks(model.layers), reg=2e-7, inverse_mask=False)
        #     train_step(images, labels, kernel_optimizer, get_all_kernels(model.layers), use_mask=False)

            # else:
            #     train_step(images1, labels1, mask_optimizer, model.trainable_variables, reg=True, )



        # for i, (images2, labels2) in enumerate(tqdm(train_ds2)):
        #     train_step(images2, labels2, mask_optimizer, get_all_masks(model.layers), reg=True)

        # else:
        #     for i, (images2, labels2) in enumerate(tqdm(train_ds2)):

        #         train_step(images2, labels2, mask_optimizer, get_all_masks(model.layers), reg=True)

        print_p_pruned(model.layers)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels, test_use_mask)



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

