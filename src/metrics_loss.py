"""Functions for loss function.
"""
# MIT License
# 
# Copyright (c) 2019 Zuheng ming
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from collections import Counter
import matplotlib.pyplot as plt
import cv2
import python_getdents 
from scipy import spatial
from sklearn.decomposition import PCA
from itertools import islice
import itertools

#import h5py


  
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
       This is not exactly the algorthim proposed in the paper, since the update/shift of the centers is not moving towards the
       centers (i.e. sum(Xi)/Nj, Xi is the element of class j) of the classes but the sum of the elements (sum(Xi)) in the class
    """
     #nrof_features = features.get_shape()[1]
     #centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
     #    initializer=tf.constant_initializer(0), trainable=False)
     #label = tf.reshape(label, [-1])
     #centers_batch = tf.gather(centers, label)
     #diff = (1 - alfa) * (centers_batch - features)
     #diff = alfa * (centers_batch - features)
     #centers = tf.scatter_sub(centers, label, diff)
    # loss = tf.nn.l2_loss(features - centers_batch)
    # return loss, centers, diff, centers_batch

    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
       -- mzh 15/02/2017
       -- Correcting the center updating, center updates/shifts towards to the center of the correponding class with a weight:
       -- centers = centers- (1-alpha)(centers-sum(Xi)/Nj), where Xi is the elements of the class j, Nj is the number of the elements of class Nj
       -- code has been tested by the test script '../test/center_loss_test.py'
    """
    with tf.variable_scope("centerloss", reuse=tf.AUTO_REUSE):
        nrof_features = features.get_shape()[1]
        # unique_label, _, _ = tf.unique_with_counts(label)
        # nrof_classes = unique_label.get_shape()[0]
        centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
        centers_cts = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
        #centers_cts_init = tf.zeros_like(nrof_classes, tf.float32)
        label = tf.reshape(label, [-1])
        centers_batch = tf.gather(centers, label) #get the corresponding center of each element in features, the list of the centers is in the same order as the features
        loss_n = tf.reduce_sum(tf.square(features - centers_batch)/2, 1)
        loss = tf.nn.l2_loss(features - centers_batch)
        diff = (1 - alfa) * (centers_batch - features)

        ## update the centers
        label_unique, idx = tf.unique(label)
        zeros = tf.zeros_like(label_unique, tf.float32)
        ## calculation the repeat time of same label
        nrof_elements_per_class_clean = tf.scatter_update(centers_cts, label_unique, zeros)
        ones = tf.ones_like(label, tf.float32)
        ## counting the number elments in each class, the class is in the order of the [0,1,2,3,....] as initialzation
        nrof_elements_per_class_update = tf.scatter_add(nrof_elements_per_class_clean, label, ones)
        ## nrof_elements_per_class_list is the number of the elements in each class in the batch
        nrof_elements_per_class_batch = tf.gather(nrof_elements_per_class_update, label)
        nrof_elements_per_class_batch_reshape = tf.reshape(nrof_elements_per_class_batch, [-1, 1])## reshape the matrix as 1 coloum no matter the dimension of the row (-1)
        diff_mean = tf.div(diff, nrof_elements_per_class_batch_reshape)
        centers = tf.scatter_sub(centers, label, diff_mean)

        #return loss, centers, label, centers_batch, diff, centers_cts, centers_cts_batch, diff_mean,center_cts_clear, nrof_elements_per_class_batch_reshape
        return loss, loss_n, centers, nrof_elements_per_class_clean, nrof_elements_per_class_batch_reshape,diff_mean # facenet_expression_addcnns_simple_joint_v4_dynamic.py
        #return loss, centers, nrof_elements_per_class_clean, nrof_elements_per_class_batch_reshape,diff_mean ### facenet_train_classifier_expression_pretrainExpr_multidata_addcnns_simple.py

def center_loss_similarity(features, label, alfa, nrof_classes):
    ## center_loss on cosine distance =1 - similarity instead of the L2 norm, i.e. Euclidian distance

    ## normalisation as the embedding vectors in order to similarity distance
    features = tf.nn.l2_normalize(features, 1, 1e-10, name='feat_emb')

    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    centers_cts = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    #centers_cts_init = tf.zeros_like(nrof_classes, tf.float32)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label) #get the corresponding center of each element in features, the list of the centers is in the same order as the features
    #loss = tf.nn.l2_loss(features - centers_batch) ## 0.5*(L2 norm)**2, L2 norm is the Euclidian distance
    similarity_all = tf.matmul(features, tf.transpose(tf.nn.l2_normalize(centers_batch, 1, 1e-10))) ## dot prodoct, cosine distance, similarity of x and y
    similarity_self = tf.diag_part(similarity_all)
    loss_x = tf.subtract(1.0, similarity_self)
    loss = tf.reduce_sum(loss_x) ## sum the cosine distance of each vector/tensor
    diff = (1 - alfa) * (centers_batch - features)
    ones = tf.ones_like(label, tf.float32)
    centers_cts = tf.scatter_add(centers_cts, label, ones) # counting the number of each class, the class is in the order of the [0,1,2,3,....] as initialzation
    centers_cts_batch = tf.gather(centers_cts, label)
    #centers_cts_batch_ext = tf.tile(centers_cts_batch, nrof_features)
    #centers_cts_batch_reshape = tf.reshape(centers_cts_batch_ext,[-1, nrof_features])
    centers_cts_batch_reshape = tf.reshape(centers_cts_batch, [-1,1])
    diff_mean = tf.div(diff, centers_cts_batch_reshape)
    centers = tf.scatter_sub(centers, label, diff_mean)
    zeros = tf.zeros_like(label, tf.float32)
    center_cts_clear = tf.scatter_update(centers_cts, label, zeros)
    #return loss, centers, label, centers_batch, diff, centers_cts, centers_cts_batch, diff_mean,center_cts_clear, centers_cts_batch_reshape
    #return loss, centers, loss_x, similarity_all, similarity_self
    return loss, centers






















  


  







  










  






















