#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 19:01:34 2018

@author: anandan
"""

import tensorflow as tf

mb_size = 100
X_dim = 1300
z_dim = 500


#with tf.name_scope('discriminator_weights'):
#
#    D_W1 = tf.Variable(tf.zeros(shape=[X_dim, 700]), name='D_W1')
#    D_b1 = tf.Variable(tf.zeros(shape=[700]), name='D_b1')
#
#    D_W2 = tf.Variable(tf.zeros(shape=[700, 200]), name='D_W2')
#    D_b2 = tf.Variable(tf.zeros(shape=[200]), name='D_b1')
#
#    D_W3 = tf.Variable(tf.zeros(shape=[200, 1]), name='D_W3')
#    D_b3 = tf.Variable(tf.zeros(shape=[1]), name='D_b3')

with tf.name_scope('generator_weigths'):

    G_W1 = tf.Variable(tf.zeros(shape=[z_dim, 700]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[700]), name='G_b1')

    G_W2 = tf.Variable(tf.zeros(shape=[700, X_dim]), name='G_w2')
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]), name='G_b2')


saver = tf.train.Saver()

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "./saved_weights/model.ckpt")
  print("Model restored.")
