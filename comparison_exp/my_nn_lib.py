#
#   my_nn_lib.py
#       date. 5/19/2016
#

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import numpy as np
# import cv2
import tensorflow as tf


# Convolution 2-D Layer
class Convolution2D(object):
    '''
      constructor's args:
          input     : input image (2D matrix)
          input_siz ; input image size
          in_ch     : number of incoming image channel
          out_ch    : number of outgoing image channel
          patch_siz : filter(patch) size
          weights   : (if input) (weights, bias)
    '''
    def __init__(self, input, input_siz, in_ch, out_ch, patch_siz, activation='relu'):
        self.input = input      
        self.rows = input_siz[0]
        self.cols = input_siz[1]
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.activation = activation

        wshape = [patch_siz[0], patch_siz[1], in_ch, out_ch]
        
        w_cv = tf.Variable(tf.truncated_normal(wshape, stddev=0.1), 
                            trainable=True)
        b_cv = tf.Variable(tf.constant(0.1, shape=[out_ch]), 
                            trainable=True)
        
        self.w = w_cv
        self.b = b_cv
        self.params = [self.w, self.b]
        
    def output(self):
        shape4D = [-1, self.rows, self.cols, self.in_ch]
        
        x_image = tf.reshape(self.input, shape4D)  # reshape to 4D tensor
        linout = tf.nn.conv2d(x_image, self.w, 
                  strides=[1, 1, 1, 1], padding='SAME') + self.b
        shapeOut = [None, self.rows, self.cols, self.out_ch]
        linout.set_shape(shapeOut)
        print("conv: ")
        print(shapeOut)                 
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(linout)
        else:
            self.output = linout
        
        return self.output

# Max Pooling Layer   
class MaxPooling2D(object):
    '''
      constructor's args:
          input  : input image (2D matrix)
          ksize  : pooling patch size
    '''
    def __init__(self, input, ksize=None):
        self.input = input
        if ksize == None:
            ksize = [1, 2, 2, 1]
            self.ksize = ksize
    
    def output(self):
        self.output = tf.nn.max_pool(self.input, ksize=self.ksize,
                    strides=[1, 2, 2, 1], padding='SAME')
        print("max pool")
        print(self.output.get_shape())
        return self.output

# Full-connected Layer   
class FullConnected(object):
    def __init__(self, input, n_in, n_out,activation='relu', name=''):
        self.input = input
    
        w_h = tf.Variable(tf.truncated_normal([n_in,n_out],
                          mean=0.0, stddev=0.05), trainable=True)
        b_h = tf.Variable(tf.zeros([n_out]), trainable=True)

        if activation == 'relu':
            print("relu")
            self.activation_fun = tf.nn.relu
        elif activation == 'tanh':
            print("tanh")
            self.activation_fun = tf.tanh
        elif activation == 'sigmoid':
            print("sigmoid")
            self.activation_fun = tf.sigmoid
        else:
            print("identity")
            self.activation_fun = tf.identity

        
        self.w = w_h
        self.b = b_h
        self.params = [self.w, self.b]
        self.name = name
    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        if(self.name!=''):
            self.output = self.activation_fun(linarg,name=self.name)
        else:
            self.output = self.activation_fun(linarg)
        print("FC")
        print(self.output.get_shape())
                        
        return self.output

# Read-out Layer
class ReadOutLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input
        
        w_o = tf.Variable(tf.random_normal([n_in,n_out],
                        mean=0.0, stddev=0.05), trainable=True)
        b_o = tf.Variable(tf.zeros([n_out]), trainable=True)

        
        self.w = w_o
        self.b = b_o
        self.params = [self.w, self.b]
    
    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.sigmoid(linarg)
        #self.output = linarg

        return self.output
#

