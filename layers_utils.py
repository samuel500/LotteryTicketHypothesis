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



masked_layers = {LotteryDense, LotteryConv2D, TrainableDropout, TrainableChannelDropout, BinaryDense, BinaryLotteryDense, BinaryLotteryConv2D}
kerneled_layers = {LotteryConv2D, LotteryDense}


class LotteryModel(Model):

    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs)
        #self.inputs = layers[0]
        self.seq_model = tf.keras.Sequential(layers)
        self.layers_list = layers

    def call(self, x, **kwargs):
        for layer in self.layers_list:
            if type(layer) in masked_layers: # careful
                x = layer(x, **kwargs)
            else:
                x = layer(x)
        return x

    def summary(self):
        self.seq_model.summary()


def get_all_masks(layers):
    masks = []
    for layer in layers:
        if type(layer) in masked_layers:
            if hasattr(layer, 'M'):
                if layer.M.trainable:
                    masks.append(layer.M)
            if hasattr(layer, 'WM'):
                if layer.WM.trainable:
                    masks.append(layer.WM)

    return masks

def get_all_kernels(layers):
    kernels = []
    for layer in layers:
        if type(layer) in kerneled_layers:
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
        if type(layer) not in masked_layers:
            if layer.trainable_variables:
                normals += layer.trainable_variables
    return normals


def print_p_pruned(layers):

    tot_w = 0
    tot_m = 0
    for i, l in enumerate(layers):
        if type(l) in masked_layers:
            tot = np.prod(l.M.shape)
            tot_w += tot
            mask = l.get_int_mask()
            #print(mask)
            m = tf.math.count_nonzero(mask).numpy()
            tot_m += m
            print('Layer', i, '('+str(type(l))+')', 'p pruned:', 1-m/tot)


    print('Tot p pruned:', 1-tot_m/tot_w)

    return 1-tot_m/tot_w

