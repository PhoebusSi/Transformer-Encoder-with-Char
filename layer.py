# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:37:55 2019

@author: jbk48
"""

import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' #use GPU with ID=0

def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf



class FFN:
    """FFN class (Position-wise Feed-Forward Networks)"""

    def __init__(self,
                 w1_dim=200,
                 w2_dim=100,
                 dropout=0.1):

        self.w1_dim = w1_dim
        self.w2_dim = w2_dim
        self.dropout = dropout

    def dense_relu_dense(self, inputs):
        output = tf.layers.dense(inputs, self.w1_dim, activation=tf.nn.relu)
        output =tf.layers.dense(output, self.w2_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def dense_gelu_dense(self, inputs):
        with tf.variable_scope("dense_gelu_dense",reuse=tf.AUTO_REUSE) as scope:
            output = tf.layers.dense(inputs, self.w1_dim, activation=gelu)
        with tf.variable_scope("dense_gelu_dense_2",reuse=tf.AUTO_REUSE) as scope:
            output =tf.layers.dense(output, self.w2_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def conv_relu_conv(self):
        raise NotImplementedError("i will implement it!")


