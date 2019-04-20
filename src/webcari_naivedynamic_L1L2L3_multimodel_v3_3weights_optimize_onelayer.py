"""Functions for navie dynamic multi-task caricature-visual image face recogntion on dataset webcaricature.
"""
# MIT License
#
# Copyright (c) 2019 Zuheng Ming

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import h5py
from sklearn.decomposition import PCA
import glob
import shutil
import matplotlib.pyplot as plt
import csv
import cv2
import math
import glob
from numpy import linalg as LA
import imp
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import debug as tf_debug
from random import shuffle
from itertools import compress


###### user custom lib
import facenet_ext
import lfw_ext
import metrics_loss
import train_BP



def main(args):
    
    module_networks = str.split(args.model_def,'/')[-1]
    network = imp.load_source(module_networks, args.model_def)  


    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet_ext.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    #########################   webcari    ##########################
    image_list, label_list_id, label_list_cari, _, nrof_classes \
        = facenet_ext.get_image_paths_and_labels_webcari(args.train_pairs, args.data_dir)

    image_list_test, label_list_id_test, label_cari_test, label_verif_test, nrof_classes_test \
        = facenet_ext.get_image_paths_and_labels_webcari(args.test_pairs, args.data_dir)
    
    ## check whether all the class in the test are included in the training set
    label_list_id_set = list(set(label_list_id))
    label_list_id_test_set = list(set(label_list_id_test))
    label_list_id_test_set_in = [x for x in label_list_id_test_set if x in label_list_id_set]
    if (len(label_list_id_test_set_in) < len(label_list_id_test_set)):
        label_list_id_all_set = list(set(label_list_id_set + label_list_id_test_set))
    else: 
        label_list_id_all_set = label_list_id_set

    label_list_id_all_set.sort()
    ## mapping the label to the number instead of the id name
    label_list_id = [label_list_id_all_set.index(x) for x in label_list_id ]
    label_list_id_test = [label_list_id_all_set.index(x) for x in label_list_id_test]

    ## filtering the visual images in test dataset
    filter = [x==0 for x in label_cari_test]

    # filtering the caricature images in test dataset
    filter = [x==1 for x in label_cari_test]


    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        meta_file, ckpt_file = facenet_ext.get_model_filenames(os.path.expanduser(args.pretrained_model))
        print('Pre-trained model: %s' % pretrained_model)

    #########################   webcari    ##########################
    if args.evaluate_express:
        print('Test data directory: %s' % args.data_dir)

        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)


        
        # Create a queue that produces indices into the image_list and label_list_id 
        labels = ops.convert_to_tensor(label_list_id, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)
        
        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        phase_train_placeholder_expression = tf.placeholder(tf.bool, name='phase_train_expression')

        phase_train_placeholder_cari = tf.placeholder(tf.bool, name='phase_train_cari')

        
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')

        labels_id_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels_id')

        labels_expr_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels_expr')

        labels_cari_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels_cari')

        keep_probability_placeholder = tf.placeholder(tf.float32, name='keep_probability')

        input_queue = data_flow_ops.FIFOQueue(max(100000, args.batch_size*args.epoch_size),
                                    dtypes=[tf.string, tf.int64, tf.int64, tf.int64,],
                                    shapes=[(1,), (1,), (1,), (1,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_id_placeholder, labels_expr_placeholder, labels_cari_placeholder], name='enqueue_op')
        
        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label_id, label_expr, label_cari = input_queue.dequeue()
            images = []
            #for filename in tf.unpack(filenames): ## tf0.12
            for filename in tf.unstack(filenames): ## tf1.0
                file_contents = tf.read_file(filename)
                image = tf.image.decode_png(file_contents)
                #image = tf.image.decode_jpeg(file_contents)
                if args.random_rotate:
                    image = tf.py_func(facenet_ext.random_rotate_image, [image], tf.uint8)
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_images(image, [args.image_size, args.image_size]) ## if input is face image, keep the whole image
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
    
                #pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                ### whiten image
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label_id, label_expr, label_cari])
    
        image_batch, label_batch_id, label_batch_expr, label_batch_cari = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size_placeholder,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            enqueue_many=True,
            shapes=[(args.image_size, args.image_size, 3), (), (), ()],
            allow_smaller_final_batch=True)
        #image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_id_batch = tf.identity(label_batch_id, 'label_id_batch')
        label_expr_batch = tf.identity(label_batch_expr, 'label_expr_batch')
        label_cari_batch = tf.identity(label_batch_cari, 'label_cari_batch')


        ################################ Branch Cari-Visual ###############################################
        print('Building training graph')
        
        # Build the inference graph
        prelogits, end_points = network.inference(image_batch, keep_probability_placeholder,
            phase_train=phase_train_placeholder, weight_decay=args.weight_decay)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        #logits_id = slim.fully_connected(prelogits, nrof_classes+nrof_classes_test, activation_fn=None, weights_initializer= tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits_verif', reuse=False)
        logits_id = slim.fully_connected(prelogits, len(label_list_id_all_set), activation_fn=None, weights_initializer= tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits_verif', reuse=False)


	
        # Add center loss
        if args.center_loss_factor>0.0:
            prelogits_center_loss_verif, prelogits_center_loss_verif_n, centers, _, centers_cts_batch_reshape, diff_mean \
                = metrics_loss.center_loss(embeddings, label_id_batch, args.center_loss_alfa, nrof_classes)

        cross_entropy_verif = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_id, labels=label_id_batch, name='cross_entropy_batch_verif')
        cross_entropy_mean_verif = tf.reduce_mean(cross_entropy_verif, name='cross_entropy_verif')

        loss_verif_n = cross_entropy_verif + args.center_loss_factor*prelogits_center_loss_verif_n
        #loss_verif_n = cross_entropy_verif
        loss_verif = tf.reduce_mean(loss_verif_n, name='loss_verif')



        inputs = end_points['Mixed_7a']

        prelogits_expression, end_points_expression = network.inference_expression(inputs, keep_probability_placeholder, phase_train=phase_train_placeholder_expression, weight_decay=args.weight_decay)
        embeddings_expression = tf.nn.l2_normalize(prelogits_expression, 1, 1e-10, name='embeddings_expression')

        mask_expr = tf.equal(label_cari_batch, 0)
        embeddings_expression_filter = tf.boolean_mask(embeddings_expression, mask_expr)
        label_id_filter_expr = tf.boolean_mask(label_id_batch, mask_expr)

        prelogits_expression_center_loss, prelogits_expression_center_loss_n, centers_expression, _, centers_cts_batch_reshape_expression, diff_mean_expression \
            = metrics_loss.center_loss(embeddings_expression_filter, label_id_filter_expr, args.center_loss_alfa, nrof_classes)


        logits_expr = slim.fully_connected(prelogits_expression, len(label_list_id_all_set), activation_fn=tf.nn.relu,weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits', reuse=False)

        logits_expr = tf.identity(logits_expr, 'logits_expr')

        ## Filtering the visual image for training the Branch Visual recognition
        
        logits_expr_filter = tf.boolean_mask(logits_expr, mask_expr)
        


        # Calculate the average cross entropy loss across the batch
        cross_entropy_expr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_expr_filter, labels=label_id_filter_expr, name='cross_entropy_per_example')
        cross_entropy_mean_expr = tf.reduce_mean(cross_entropy_expr, name='cross_entropy_expr')
        #tf.add_to_collection('losses', cross_entropy_mean)
        
        # Calculate the total losses
        loss_expr_n = cross_entropy_expr+args.center_loss_factor*prelogits_expression_center_loss_n
        loss_expr = tf.reduce_mean(loss_expr_n, name='loss_expr')



        ####################### Branch for caricature recognition  ############################
        inputs = end_points['Mixed_7a']
        prelogits_cari, end_points_cari = network.inference_cari(inputs, keep_probability_placeholder,
                                                                                   phase_train=phase_train_placeholder_cari,
                                                                                   weight_decay=args.weight_decay)
        embeddings_cari = tf.nn.l2_normalize(prelogits_cari, 1, 1e-10, name='embeddings_cari')

        mask_cari = tf.equal(label_cari_batch, 1)
        embeddings_cari_filter = tf.boolean_mask(embeddings_cari, mask_cari)
        label_id_filter_cari = tf.boolean_mask(label_id_batch, mask_cari)

        prelogits_cari_center_loss, prelogits_cari_center_loss_n, centers_cari, _, centers_cts_batch_reshape_cari, diff_mean_cari \
            = metrics_loss.center_loss(embeddings_cari_filter, label_id_filter_cari, args.center_loss_alfa, nrof_classes)
        logits_cari = slim.fully_connected(prelogits_cari, len(label_list_id_all_set), activation_fn=tf.nn.relu,
                                           weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                           weights_regularizer=slim.l2_regularizer(args.weight_decay), scope='Logits_cari',
                                           reuse=False)




        logits_cari = tf.identity(logits_cari, 'logits_cari')

        ## Filtering the caricature for training the Branch Caricature recognition
        logits_cari_filter = tf.boolean_mask(logits_cari, mask_cari)
        

        cross_entropy_cari = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_cari_filter, labels=label_id_filter_cari, name='cross_entropy_per_example')
        cross_entropy_mean_cari = tf.reduce_mean(cross_entropy_cari, name='cross_entropy_cari')
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # loss_cari_n = cross_entropy_cari
        loss_cari_n = cross_entropy_cari+args.center_loss_factor*prelogits_cari_center_loss_n        
        loss_cari = tf.reduce_mean(loss_cari_n, name='loss_cari')

        ##################################################################################################################
        ##### Dynamic weight module to learn the weight of the facial expression loss and the face verif weight in full loss ####
        ##################################################################################################################
        input_lossweights = slim.flatten(inputs)
        input_lossweights_embedings = tf.nn.l2_normalize(input_lossweights, 1, 1e-10, name='norm_lossweights')

        #layer1_lossweights = slim.fully_connected(input_lossweights, 512, activation_fn=tf.nn.relu,weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Layer_lossweights', reuse=False)
        ###### Normalization the layer1_lossweights to evoid inf of logits_lossweights ##############
        #layer1_lossweights_embedings = tf.nn.l2_normalize(layer1_lossweights, 1, 1e-10, name='norm_lossweights')


        ##################################################################################################################
        ############### Attention, logits_lossweights layer Not use RELU Activative function, since all the minor values
        ############### will be set to 0; this will eaisly make the values the logits_lossweights to be zero for all the images in the batch
        ##################################################################################################################
        logits_lossweights = slim.fully_connected(input_lossweights_embedings, 3, activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits_lossweights', reuse=False)
        ###### Normalization the layer1_lossweights to evoid inf of exp(logits_lossweights) ##############
        logits_lossweights_embedings = tf.nn.l2_normalize(logits_lossweights, 1, 1e-10, name='norm_lossweights')/2
        #logits_lossweights_embedings = logits_lossweights_embedings * 2 ## embedings [0-1] is too small
        softmax_lossweights =tf.nn.softmax(logits_lossweights_embedings)
        #softmax_lossweights =tf.nn.softmax(logits_lossweights)
        loss_verif_percentage = tf.reduce_mean(softmax_lossweights[:, 0])+args.loss_weight_base1
        loss_expr_percentage = tf.reduce_mean(softmax_lossweights[:, 1])+args.loss_weight_base2
        loss_cari_percentage = tf.reduce_mean(softmax_lossweights[:, 2])+args.loss_weight_base3
        #loss_cari_percentage = 1.0

        ###################### alpha weights optimization loss  ##########################
        sigma = 0.01
        loss_for_weights = tf.add_n([tf.divide(loss_verif_percentage, loss_verif+sigma)]
                                           + [tf.divide(loss_expr_percentage, loss_expr+sigma)]
                                           + [tf.divide(loss_cari_percentage, loss_cari+sigma)],
                                           name='loss_for_weights')

        ##################### Full loss ###################################
        # loss_verif_percentage = tf.cast(0.55, tf.float32)
        # loss_expr_percentage = tf.cast(0.30, tf.float32)
        # loss_cari_percentage = tf.cast(0.15, tf.float32)

        weight_fullcrossentropy = tf.add_n([tf.multiply(loss_verif_percentage, loss_verif)]
                                            +[tf.multiply(loss_expr_percentage, loss_expr)]
                                           +[tf.multiply(loss_cari_percentage, loss_cari)],
                                           name='weight_fullcrossentropy')


        # weight_fullcrossentropy = tf.add_n([tf.multiply(loss_verif_percentage, loss_verif)]
        #                                    +[tf.multiply(loss_expr_percentage, loss_expr)]
        #                                    +[tf.multiply(loss_cari_percentage, loss_cari)],
        #                                    name='weight_fullcrossentropy')

        # weight_fullcrossentropy = tf.add_n([loss_cari],
        #                                    name='weight_fullcrossentropy')

        #weight_fullcrossentropy = tf.add_n([tf.multiply(loss_verif_percentage+args.loss_weight_base, loss_verif_n)]+[tf.multiply(loss_expr_percentage, loss_expr_n)], name='weight_fullcrossentropy')
        loss_full = tf.reduce_mean(weight_fullcrossentropy, name='loss_full')



        # #### Training accuracy of softmax: check the underfitting or overfiting #############################
        correct_prediction_verif = tf.equal(tf.argmax(tf.exp(logits_id), 1), label_batch_id)
        softmax_acc_verif = tf.reduce_mean(tf.cast(correct_prediction_verif, tf.float32))
        #correct_prediction_expr = tf.equal(tf.argmax(tf.exp(logits_expr), 1), label_batch_expr)
        #correct_prediction_cari = tf.equal(tf.argmax(tf.exp(logits_cari), 1), label_batch_cari)
        correct_prediction_expr = tf.equal(tf.argmax(tf.exp(logits_expr_filter), 1), label_id_filter_expr)
        correct_prediction_cari = tf.equal(tf.argmax(tf.exp(logits_cari_filter), 1), label_id_filter_cari)
        softmax_acc_expr = tf.reduce_mean(tf.cast(correct_prediction_expr, tf.float32))
        softmax_acc_cari = tf.reduce_mean(tf.cast(correct_prediction_cari, tf.float32))
        ########################################################################################################

        ###### Automatic lower learning rate lr= lr_start * decay_factor^(global_step/decaystepwidth),
        ###### if decay_factor = 1.0 it means the learning rate will not decay automaically, otherwise it will decay
        ###### from the given learning rate in function of the factor, current global step and decay ste width.
        if args.learning_rate>0.0:
            # learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            #                                            args.learning_rate_decay_epochs*args.epoch_size,
            #                                            args.learning_rate_decay_factor, staircase=True)
            learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                        args.learning_rate_decay_epochs*args.epoch_size,
                                                        1.0, staircase=True)
        else:
            learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                       args.learning_rate_decay_epochs * args.epoch_size,
                                                       1.0, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        update_gradient_vars_expr = []
        update_gradient_vars_verif = []


        update_gradient_vars_mainstem = tf.trainable_variables()
        paracnt, parasize = count_paras(update_gradient_vars_verif)
        print('The number of the updating parameters in the model Facenet is %dM, ......the size is : %dM bytes'%(paracnt/1e6, parasize/1e6))

        paracnt, parasize = count_paras(update_gradient_vars_expr)
        print('The number of the update parameters in the model Facial Expression is %dM, ......the size is : %dM bytes'%(paracnt/1e6, parasize/1e6))

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()


        train_op_mainstem, grads_full, grads_clip_full = train_BP.train(loss_full, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, update_gradient_vars_mainstem, summary_op, args.log_histograms)
        # train_op_weights, grads_weights, grads_clip__weights = train_BP.train(loss_for_weights, global_step, args.optimizer,
        #     learning_rate, args.moving_average_decay, update_gradient_vars_weights, summary_op, args.log_histograms)


        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        #sess = tf.Session(config=tf.ConfigProto(device_count={'CPU':1}, log_device_placement=True))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        tf.train.start_queue_runners(sess=sess) ## wakeup the queue: start the queue operating defined by the tf.train.batch_join

        ### debug tfdeb #####
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess, thread_name_filter="MainThread$")
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        ### debug tfdeb #####

        with sess.as_default():

            if pretrained_model:
                reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
                var_to_shape_map = reader.get_variable_to_shape_map()
                i=0
                isExpressionModel = False
                isCariModel = False
                for key in sorted(var_to_shape_map):
                    print(key)
                    ##xx= reader.get_tensor(key)
                    if 'InceptionResnetV1_expression' in key:
                        isExpressionModel = True
                    if 'InceptionResnetV1_cari' in key:
                        isCariModel = True
                    i += 1
                print('################## %d parametrs in the pretrained model ###################'%i)

                restore_vars = []
                if isCariModel:
                    restore_saver_cavi = tf.train.Saver(tf.global_variables())
                    restore_saver_cavi.restore(sess, os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
                if isExpressionModel:
                    print('>>>>>>>>>>>> Loading directly the pretrained FaceLiveNet model :%s.....'% os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))

                    restore_vars_verif = []
                    restore_vars_expre = []
                    ## load the face_verif networks parameters
                    #for var in tf.trainable_variables():
                    for var in tf.global_variables():
                        if 'InceptionResnetV1/' in var.op.name and args.optimizer not in var.op.name and 'ExponentialMovingAverage' not in var.op.name:
                            #if 'center' not in var.op.name and 'Logits_verif/' not in var.op.name:
                            restore_vars_verif.append(var)
                        else:
                            print(var.op.name)

                    ## load the face_expression networks parameters
                    for var in tf.global_variables():
                        ##corresponding to the variable global_step saved in the pretrained model
                         if 'Variable' in var.op.name:
                             restore_vars_expre.append(var)
                         ##corresponding to the variable of expression model
                         #if 'InceptionResnetV1_expression/' in var.op.name or 'Logits/' in var.op.name or 'Logits_0/' in var.op.name:
                         if 'InceptionResnetV1_expression/' in var.op.name or 'Logits_0/' in var.op.name:
                             restore_vars_expre.append(var)
                    paracnt_verif, parasize_verif = count_paras(restore_vars_verif)
                    paracnt_expr, parasize_expr = count_paras(restore_vars_expre)
                    paracnt = paracnt_verif + paracnt_expr
                    parasize = parasize_verif +  parasize_expr
                    print('The number of the loading parameters in the model(FaceLiveNet) is %dM, ......the size is : %dM bytes, the Face_verif networks is %dM, ......the size is : %dM bytes, the Face_expression networks is %dM, ......the size is : %dM bytes'
                          % (paracnt / 1e6, parasize / 1e6, paracnt_verif / 1e6, parasize_verif / 1e6, paracnt_expr / 1e6, parasize_expr / 1e6))
                    restore_saver_verif = tf.train.Saver(restore_vars_verif)
                    restore_saver_verif.restore(sess, os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
                    restore_saver_expression = tf.train.Saver(restore_vars_expre)
                    restore_saver_expression.restore(sess, os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))

                    # restore_vars_global = tf.global_variables()
                    # restore_saver_global = tf.train.Saver(restore_vars_global)
                    # restore_saver_global.restore(sess, os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
                else:
                    restore_vars_verif = []
                    restore_vars_expre = []
                    print('Restoring pretrained FaceNet model: %s' % os.path.join(os.path.expanduser(args.pretrained_model),
                                                                          ckpt_file))
                    # saver.restore(sess, pretrained_model)
                    for var in tf.global_variables():
                    #for var in restore_vars:
                        if 'InceptionResnetV1/' in var.op.name:
                            restore_vars_verif.append(var)
                        if 'InceptionResnetV1_expression/' in var.op.name:
                            restore_vars_expre.append(var)
                    paracnt, parasize = count_paras(restore_vars_verif)
                    print('The number of the loading parameters in the model(FaceNet) is %dM, ......the size is : %dM bytes' % (
                            paracnt / 1e6, parasize / 1e6))

                    #saver.restore(sess, pretrained_model)
                    restore_saver = tf.train.Saver(restore_vars_verif)
                    restore_saver.restore(sess, os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
                    j = 0
                    for i, var in enumerate(restore_vars_expre):
                        var_name = str.split(str(var.op.name), 'InceptionResnetV1_expression/')[1]
                        #### only used for the block8 branch cut case ##########
                                #if 'block8_branchcut_1/Conv2d_1x1' in var_name:
                        #	continue
                        #### only used for the block8 branch cut case ##########
                        if 'Repeat' in var_name:
                            var_name = str.split(var_name,'Repeat')[1]
                            pos = var_name.index('/')
                            var_name = var_name[pos:]
                            if 'block8_branchcut_1/' in var_name:
                                var_name = str.split(var_name,'block8_branchcut_1/')[1]
                                var_name = 'block8_1/'+var_name
                            #var_name = 'block8_1'+var_name[1:]
                            #var_name = 'block8_'+var_name
                        #for var0 in restore_vars:
                        for var0 in restore_vars_verif:
                            if var_name in str(var0.op.name):
                                sess.run(var.assign(var0))
                                print(j)
                                var0_sum = np.sum(sess.run(var0))
                                var_sum = np.sum(sess.run(var))
                                print(var0.op.name, '===========>>>>>>>>> var0_sum:%f'%var0_sum)
                                print(var.op.name, '===========>>>>>>>>>  var_sum:%f' %var_sum)
                                if var0_sum != var_sum:
                                    raise ValueError('Error of the assignment form var0 to var!')
                                j += 1
                                break

            # Training and validation loop
            # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
            #saver = tf.train.Saver(restore_vars, max_to_keep=2)

            print('Running training')
            epoch = 0
            acc = 0
            val = 0
            far = 0
            acc_expr_paris = 0
            val_expr_paris = 0
            far_expr_paris = 0
            best_acc_exprecog = 0
            best_acc_faceverif_expr = 0
            best_acc_faceverif_lfw = 0
            best_acc_faceauthen = 0
            best_authen_verif_exprpairs = 0
            best_authen_exprecog = 0
            acc_expression = 0
            acc_cari = 0
            acc_c2v = 0
            acc_v2c = 0
            acc_mixrecog = 0

            best_acc_carirecog = 0
            best_acc_c2v = 0
            best_acc_v2c = 0
            best_acc_mixrecog = 0

            nrof_expressions = len(set(label_list_id))
            each_expr_acc = np.zeros(nrof_expressions)
            express_probs_confus_matrix = np.zeros((nrof_expressions,nrof_expressions))

            epoch_current = 0
            f_weights = open(os.path.join(log_dir, 'percentage_full_loss.txt'), 'at')
            f_weights.write('loss_verif_weight, loss_expr_weight, loss_cari_weight\n')
            f_loss = open(os.path.join(log_dir, 'loss.txt'), 'at')
            f_loss.write('loss_verif, loss_expr, loss_cari, cross_entropy_mean_verif, cross_entropy_mean_expr, \
                cross_entropy_mean_cari, prelogits_center_loss_verif\n')

            while epoch < args.max_nrof_epochs:
                epoch_current +=1
                step = sess.run(global_step, feed_dict=None)
                print('Epoch step: %d'%step)
                epoch = step // args.epoch_size
                # Train for one epoch
                step, softmax_acc_verif_, softmax_acc_expr_, loss_verif_, \
                loss_expr_, cross_entropy_mean_verif_, cross_entropy_mean_expr_, Reg_loss,  center_loss_, verifacc, \
                learning_rate_, loss_verif_weight, loss_expr_weight, loss_cari_weight, loss_cari_, cross_entropy_mean_cari_, prelogits_center_loss_verif_\
                    = train(args, sess, epoch, image_list, label_list_id, index_dequeue_op, enqueue_op,
                            image_paths_placeholder, labels_id_placeholder, labels_expr_placeholder,
                            learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                            loss_verif, loss_expr, summary_op, summary_writer,
                            regularization_losses, args.learning_rate_schedule_file, prelogits_center_loss_verif,
                            cross_entropy_mean_verif, cross_entropy_mean_expr, acc, val, far, centers_cts_batch_reshape,
                            logits_id, logits_expr, keep_probability_placeholder, update_gradient_vars_expr, acc_expression,
                            each_expr_acc, label_batch_id, label_batch_expr, express_probs_confus_matrix, log_dir,
                            model_dir, image_batch, learning_rate, phase_train_placeholder_expression,
                            best_acc_exprecog, softmax_acc_verif, softmax_acc_expr, cross_entropy_verif, diff_mean,
                            centers, acc_expr_paris, val_expr_paris, far_expr_paris, best_acc_faceverif_expr,
                            best_acc_faceverif_lfw, train_op_mainstem, best_acc_faceauthen, best_authen_verif_exprpairs,
                            best_authen_exprecog, loss_verif_percentage, loss_expr_percentage, epoch_current, logits_lossweights_embedings,
                            label_list_cari, labels_cari_placeholder, phase_train_placeholder_cari, softmax_acc_cari,
                            loss_cari, cross_entropy_mean_cari, loss_cari_percentage, acc_cari, best_acc_carirecog,
                            acc_v2c, acc_c2v, best_acc_c2v, best_acc_v2c, acc_mixrecog, best_acc_mixrecog,
                            loss_for_weights, f_weights, f_loss)


                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)


                if (epoch % 1 == 0):
                     if args.expr_pairs:
                         acc_expr_paris, val_expr_paris, far_expr_paris, acc_mixrecog = evaluate(sess, enqueue_op, image_paths_placeholder, labels_id_placeholder,
                                                  labels_expr_placeholder, phase_train_placeholder, batch_size_placeholder,embeddings, label_id_batch,
                                                  image_list_test, label_verif_test, args.lfw_batch_size, args.lfw_nrof_folds,
                                                  log_dir, step, summary_writer, args.evaluate_mode,
                                                  keep_probability_placeholder, 'cari-verif-pairs', best_acc_faceverif_expr,
                                                  args, logits_id, label_list_id_test)



                ## saving the best_model for cari-visual verification
                if acc_expr_paris > best_acc_faceverif_expr:
                    best_acc_faceverif_expr = acc_expr_paris
                    best_model_dir = os.path.join(model_dir, 'best_model_verifexpr')
                    if not os.path.isdir(best_model_dir):  # Create the log directory if it doesn't exist
                        os.makedirs(best_model_dir)
                    if os.listdir(best_model_dir):
                        for file in glob.glob(os.path.join(best_model_dir, '*.*')):
                            os.remove(file)
                    for file in glob.glob(os.path.join(model_dir, '*.*')):
                        shutil.copy(file, best_model_dir)


                if acc_expression>best_acc_exprecog:
                    best_acc_exprecog = acc_expression
                    best_model_dir = os.path.join(model_dir, 'best_model_expr')
                    if not os.path.isdir(best_model_dir):  # Create the log directory if it doesn't exist
                        os.makedirs(best_model_dir)
                    if os.listdir(best_model_dir):
                        for file in glob.glob(os.path.join(best_model_dir, '*.*')):
                            os.remove(file)
                    for file in glob.glob(os.path.join(model_dir, '*.*')):
                        shutil.copy(file, best_model_dir)



                ## saving the best_model for cari recognition
                if acc_cari > best_acc_carirecog:
                    best_acc_carirecog = acc_cari
                    best_model_dir = os.path.join(model_dir, 'best_model_cari')
                    if not os.path.isdir(best_model_dir):  # Create the log directory if it doesn't exist
                        os.makedirs(best_model_dir)
                    if os.listdir(best_model_dir):
                        for file in glob.glob(os.path.join(best_model_dir, '*.*')):
                            os.remove(file)
                    for file in glob.glob(os.path.join(model_dir, '*.*')):
                        shutil.copy(file, best_model_dir)

                ## the best_performance for c2v recognition
                if acc_c2v > best_acc_c2v:
                    best_acc_c2v = acc_c2v

                ## the best_performance for v2c recognition
                if acc_v2c > best_acc_v2c:
                    best_acc_v2c = acc_v2c

                ## the best_performance for mix recognition
                if acc_mixrecog > best_acc_mixrecog:
                    best_acc_mixrecog = acc_mixrecog



    return model_dir
  
def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold
  

  
def train(args, sess, epoch, image_list, label_list_id, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_id_placeholder, labels_expr_placeholder, learning_rate_placeholder, phase_train_placeholder,
          batch_size_placeholder, global_step, loss_verif, loss_expr, summary_op,
          summary_writer, regularization_losses, learning_rate_schedule_file, prelogits_center_loss_verif,
          cross_entropy_mean_verif, cross_entropy_mean_expr, acc, val, far, centers_cts_batch_reshape, logits_id,
          logits_expr, keep_probability_placeholder, update_gradient_vars_expr, acc_expression, each_expr_acc,
          label_batch_id, label_batch_expr, express_probs_confus_matrix, log_dir, model_dir,
          image_batch, learning_rate, phase_train_placeholder_expression, best_acc_exprecog, softmax_acc_verif,
          softmax_acc_expr, cross_entropy_verif, diff_mean, centers, acc_expr_paris, val_expr_paris, far_expr_paris,
          best_acc_faceverif_expr, best_acc_faceverif_lfw, train_op_mainstem, best_acc_faceauthen, best_authen_verif_exprpairs,
          best_authen_exprecog, loss_verif_percentage, loss_expr_percentage, epoch_current, logits_lossweights_embedings, label_list_cari,
          labels_cari_placeholder, phase_train_placeholder_cari, softmax_acc_cari, loss_cari, cross_entropy_mean_cari,
          loss_cari_percentage, acc_cari, best_acc_carirecog, acc_v2c, acc_c2v, best_acc_c2v, best_acc_v2c, acc_mixrecog,
          best_acc_mixrecog, loss_for_weights, f_weights, f_loss):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet_ext.get_learning_rate_from_file(learning_rate_schedule_file, epoch_current)

    print('Index_dequeue_op....')
    index_epoch = sess.run(index_dequeue_op)
    label_id_epoch = np.array(label_list_id)[index_epoch]
    #label_expr_epoch = np.array(label_list_id)[index_epoch]
    label_cari_epoch = np.array(label_list_cari)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    # for i in range(10):
    #     print('index: %d' %index_epoch[i])
    #     print('label_epoch: %d,%d image_epoch: %s' % (label_epoch[i][0],label_epoch[i][1], image_epoch[i]))



    
    print('Enqueue__op....')
    # Enqueue one epoch of image paths and labels
    labels_id_array = np.expand_dims(np.array(label_id_epoch),1)
    #labels_expr_array = np.expand_dims(np.array(label_expr_epoch), 1)
    labels_cari_array = np.expand_dims(np.array(label_cari_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    #sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_id_placeholder: labels_id_array, labels_expr_placeholder: labels_expr_array})
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_id_placeholder: labels_id_array,
                          labels_expr_placeholder: labels_id_array, labels_cari_placeholder: labels_cari_array,})

    # Training loop
    train_time = 0

    ####################### summing up the values the dimensions of the variables for checking the updating of the variables ##########################
    vars_ref = []
    check_vars = update_gradient_vars_expr
    #check_vars = tf.trainable_variables()
    for var in check_vars:
    #for var in tf.trainable_variables():
        var_value = sess.run(var)
        vars_ref.append(np.sum(var_value))
    ####################### summing up the values the dimensions of the variables for checking the updating of the variables ##########################
    # Add validation loss and accuracy to summary
    summary = tf.Summary()

    while batch_number < args.epoch_size:
        start_time = time.time()
        #feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size, keep_probability_placeholder: args.keep_probability}
        feed_dict = {learning_rate_placeholder: lr,
                     phase_train_placeholder: True,
                     phase_train_placeholder_expression: True,
                     phase_train_placeholder_cari: True,
                     batch_size_placeholder: args.batch_size,
                     keep_probability_placeholder: args.keep_probability}
        if (batch_number % args.epoch_size == 0):

            loss_verif_, loss_expr_, step, reg_loss, prelogits_center_loss_verif_, cross_entropy_mean_verif_, \
            cross_entropy_mean_expr_, centers_cts_batch_reshape_, logits_id_, logits_expr_, label_batch_id_, \
            label_batch_expr_, image_batch_, learning_rate_, softmax_acc_verif_, \
            softmax_acc_expr_, cross_entropy_verif_, diff_mean_, centers_, _, loss_verif_percentage_, \
            loss_expr_percentage_,logits_lossweights_embedings_, softmax_acc_cari_, loss_cari_, cross_entropy_mean_cari_, loss_cari_percentage_,\
            loss_for_weights_, summary_str \
                = sess.run([loss_verif, loss_expr, global_step, regularization_losses,
                            prelogits_center_loss_verif, cross_entropy_mean_verif, cross_entropy_mean_expr,
                            centers_cts_batch_reshape, logits_id, logits_expr, label_batch_id, label_batch_expr,
                            image_batch, learning_rate, softmax_acc_verif, softmax_acc_expr,
                            cross_entropy_verif, diff_mean, centers, train_op_mainstem, loss_verif_percentage,
                            loss_expr_percentage, logits_lossweights_embedings, softmax_acc_cari, loss_cari,
                            cross_entropy_mean_cari, loss_cari_percentage, loss_for_weights,
                            summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            loss_verif_, loss_expr_, step, reg_loss, prelogits_center_loss_verif_, cross_entropy_mean_verif_, \
            cross_entropy_mean_expr_, centers_cts_batch_reshape_, logits_id_, logits_expr_, label_batch_id_, \
            label_batch_expr_, image_batch_, learning_rate_, softmax_acc_verif_, \
            softmax_acc_expr_, cross_entropy_verif_, diff_mean_, centers_, _, loss_verif_percentage_, \
            loss_expr_percentage_, logits_lossweights_embedings_, softmax_acc_cari_, loss_cari_, cross_entropy_mean_cari_, \
            loss_cari_percentage_,loss_for_weights_\
                = sess.run([loss_verif, loss_expr, global_step, regularization_losses,
                            prelogits_center_loss_verif, cross_entropy_mean_verif, cross_entropy_mean_expr,
                            centers_cts_batch_reshape, logits_id, logits_expr, label_batch_id, label_batch_expr,
                            image_batch, learning_rate, softmax_acc_verif, softmax_acc_expr,
                            cross_entropy_verif, diff_mean, centers, train_op_mainstem, loss_verif_percentage,
                            loss_expr_percentage, logits_lossweights_embedings, softmax_acc_cari, loss_cari,
                            cross_entropy_mean_cari, loss_cari_percentage, loss_for_weights], feed_dict=feed_dict)
        print("step %d"%step)
        duration = time.time() - start_time

        ##### saving the weights of loss and the loss
        f_weights.write('%2.4f %2.4f %2.4f\n' % (loss_verif_percentage_, loss_expr_percentage_, loss_cari_percentage_))
        f_loss.write('%2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f\n' % (
            loss_verif_, loss_expr_, loss_cari_, cross_entropy_mean_verif_, cross_entropy_mean_expr_,
            cross_entropy_mean_cari_, prelogits_center_loss_verif_))


        ####################### check the state of the update weights/bias in the variables by summing up the values the dimensions of the variables ##########################
        vars_sum = 0
        ii = 0
        dd = 0
        parasize = 0
        paracnt = 0
        for var in check_vars:
            var_value = sess.run(var)
            if np.sum(var_value) != vars_ref[ii]:
                #print(var.op.name)
                dd += 1
                #raw_input()
            vars_sum += np.sum(var_value)
            ii +=1

            paranum = 1
            for dim in var.get_shape():
                paranum *= dim.value
            parasize += paranum * sys.getsizeof(var.dtype)
            paracnt += paranum
        print('%d vars changed'%dd)
        print('The number of the update parameters in the model is %dM, ......the size is : %dM bytes' % (paracnt / 1e6, parasize / 1e6))
        ####################### check the state of the update weights/bias in the variables by summing up the values the dimensions of the variables ##########################



        print('########## log dir is: %s '%log_dir )
        print('########## model dir is: %s ' %model_dir)
        # print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.4f\tCrossEntropy %2.4f\tRegLoss %2.4f\tCenterLoss_cosine %2.4f\tAcc %2.4f\tVal %2.4f\tFar %2.4f' %
        #       (epoch, batch_number+1, args.epoch_size, duration, err, cross_entropy_mean_, np.sum(reg_loss), prelogits_center_loss_verif_, acc, val, far))
        print('Epoch: [%d][%d][%d/%d] Time %.3f\n \
        Loss_verif %2.4f Loss_visual-recog %2.4f Loss_cari-recog %2.4f CrossEntropy_verif %2.4f CrossEntropy_visual-recog %2.4f CrossEntropy_cari-recog %2.4f\n \
        weight_verif: %2.4f weight_visual-recog: %2.4f weight_cari-recog: %2.4f  exp(logits_lossweights_embedding): %f %f %f loss_for_weights: %f \n \
        RegLoss %2.4f CenterLoss-verif_l2 %2.4f softmaxAcc_cari-visual %2.4f softmaxAcc_visual-id %2.4f softmaxAcc_cari-id %2.4f lr: %e\n' %
            (epoch, epoch_current, batch_number + 1, args.epoch_size, duration, loss_verif_, loss_expr_, loss_cari_, cross_entropy_mean_verif_, \
             cross_entropy_mean_expr_, cross_entropy_mean_cari_, np.mean(loss_verif_percentage_), np.mean(loss_expr_percentage_), np.mean(loss_cari_percentage_), \
             np.exp(np.mean(logits_lossweights_embedings_[:,0])), np.exp(np.mean(logits_lossweights_embedings_[:,1])), np.exp(np.mean(logits_lossweights_embedings_[:,2])), loss_for_weights_, \
             np.sum(reg_loss), prelogits_center_loss_verif_, softmax_acc_verif_, softmax_acc_expr_, softmax_acc_cari_, learning_rate_))

        print('Verification on Cari-Visual pairs: acc %f, val %f, far %f, best_acc %f'%(acc_expr_paris, val_expr_paris, far_expr_paris, best_acc_faceverif_expr))
        print('Visual image recognition : test_visual_acc %2.4f, best_visual_acc %2.4f'%(acc_expression, best_acc_exprecog))
        print('Caricature image recognition : test_cari_acc %2.4f, best_cari_acc %2.4f'%(acc_cari, best_acc_carirecog))
        print('C2V image recognition : test_c2v_acc %2.4f, best_c2v_acc %2.4f'%(acc_c2v, best_acc_c2v))
        print('V2C image recognition : test_v2c_acc %2.4f, best_v2c_acc %2.4f'%(acc_v2c, best_acc_v2c))
        print('Mix image recognition : test_mix_acc %2.4f, best_mix_acc %2.4f' % (acc_mixrecog, best_acc_mixrecog))




        batch_number += 1
        train_time += duration

        #summary.value.add(tag='gradient/grad_total_norm', simple_value=grad_clip_sum)
        summary_writer.add_summary(summary, step)
            
            
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    #return step, train_each_expr_acc, softmax_acc_verif_, softmax_acc_expr_, loss_verif_, loss_expr_, cross_entropy_mean_verif_, cross_entropy_mean_expr_, np.sum(reg_loss),  prelogits_center_loss_verif_, acc, learning_rate_
    return step, softmax_acc_verif_, softmax_acc_expr_, loss_verif_, loss_expr_, cross_entropy_mean_verif_, cross_entropy_mean_expr_, np.sum(reg_loss),  prelogits_center_loss_verif_, acc, learning_rate_, np.mean(loss_verif_percentage_), np.mean(loss_expr_percentage_), np.mean(loss_cari_percentage_), loss_cari_, cross_entropy_mean_cari_, prelogits_center_loss_verif_



def evaluate(sess, enqueue_op, image_paths_placeholder, labels_id_placeholder, labels_expr_placeholder, phase_train_placeholder, batch_size_placeholder, 
        embeddings, label_id_batch, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer,
             evaluate_mode, keep_probability_placeholder, dataset, best_acc, args, logits_id, label_list_test):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Evaluating face verification on '+dataset+'...')
    nrof_images = len(actual_issame) * 2
    nrof_batches = int(nrof_images / batch_size) ##floor division
    nrof_enque = batch_size*nrof_batches

    actual_issame = actual_issame[0:int(nrof_enque/2)]##left the elements in the final batch if it is not enough

    
    embedding_size = embeddings.get_shape()[1]

    emb_array = np.zeros((nrof_enque, embedding_size))
    lab_array = np.zeros((nrof_enque,))
    logits_array = np.zeros((nrof_enque, logits_id.get_shape()[1]))

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    for ii in range(nrof_batches):
        print('batch %s%d'%(dataset, ii))
        start_index = ii* batch_size
        end_index = min((ii + 1) * batch_size, nrof_images)
        paths_batch = image_paths[start_index:end_index]
        images = facenet_ext.load_data(paths_batch, False, False, args.image_size)

        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size, keep_probability_placeholder: 1.0, images_placeholder: images}
        # emb = sess.run(embeddings, feed_dict=feed_dict)
        emb, logits_ = sess.run([embeddings, logits_id], feed_dict=feed_dict)

        emb_array[start_index:end_index, :] = emb
        logits_array[start_index:end_index, :] = logits_

    id_actual = label_list_test[0:nrof_enque]
    id_probs = np.exp(logits_array) / np.tile(np.reshape(np.sum(np.exp(logits_array), 1), (logits_array.shape[0], 1)), (1, logits_array.shape[1]))
    nrof_id = id_probs.shape[1]
    id_predict = np.argmax(id_probs, 1)
    correct_prediction = np.equal(id_predict, id_actual)
    test_id_acc = np.mean(correct_prediction)

    # assert np.array_equal(lab_array, np.arange(nrof_enque))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    if evaluate_mode == 'Euclidian':
        _, _, accuracy, val, val_std, fp_idx, fn_idx,best_threshold = lfw_ext.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds, far=args.far)
    if evaluate_mode == 'similarity':
        pca = PCA(n_components=128)
        pca.fit(emb_array)
        emb_array_pca = pca.transform(emb_array)
        _, _, accuracy, val, val_std, fp_idx, fn_idx,best_threshold = lfw_ext.evaluate_cosine(emb_array_pca, actual_issame, nrof_folds=nrof_folds, far=args.far)


    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, args.far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag=dataset+'/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag=dataset+'/val_rate', simple_value=val)
    summary.value.add(tag=dataset + '/far_rate', simple_value=args.far)
    summary.value.add(tag='time/'+dataset, simple_value=lfw_time)
    summary_writer.add_summary(summary, step)

    acc = np.mean(accuracy)
    if acc > best_acc:
        np.save(os.path.join(log_dir, 'features_cari-visual-verif_emb.npy'), emb_array)
        np.save(os.path.join(log_dir, 'features_cari-visual-verif_label.npy'), id_actual)
        np.save(os.path.join(log_dir, 'features_cari-visual-verif_pairlabel.npy'), actual_issame)

    with open(os.path.join(log_dir,dataset+'_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\t%.5f\t%.5f\t%f\n' % (step, acc, val, args.far, best_acc, test_id_acc))



    return acc, val, args.far, test_id_acc



def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)

def count_paras(vars):
    parasize = 0
    paracnt = 0
    for var in vars:
        print(var)
        paranum = 1
        for dim in var.get_shape():
            paranum *= dim.value
        parasize += paranum * sys.getsizeof(var.dtype)
        paracnt += paranum

    return paracnt, parasize

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='/data/zming/datasets/CK+/CK+_mtcnnpy_182_160_expression')
    parser.add_argument('--data_dir_test', type=str,
        help='Path to the data directory containing aligned face for test. Multiple directories are separated with colon.')
    parser.add_argument('--labels_expression', type=str,
        help='Path to the Emotion labels file.', default='~/datasets/zming/datasets/CK+/Emotion_labels.txt')
    parser.add_argument('--labels_expression_test', type=str,
        help='Path to the Emotion labels file for test.', default='~/datasets/zming/datasets/CK+/Emotion_labels.txt')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.nn4')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--decov_loss_factor', type=float,
        help='DeCov loss factor.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['Adagrad', 'Adadelta', 'Adam', 'RMSProp', 'Momentum', 'SGD'],
        help='The optimization algorithm to use', default='RMSProp')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=0.1)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augumentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='../data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--evaluate_express', type=bool,
                        help='Whether having the valdating process during the training', default=True)
    parser.add_argument('--augfer2013',
                       help='Whether doing the augmentation for the FER2013 dataset to balance the classes', action='store_true')
    parser.add_argument('--nfold', type=int,
                        help='The ith fold used in the n-fold cross-validation', default=10)
    parser.add_argument('--ifold', type=int,
                        help='The ith fold used in the n-fold cross-validation', default=0)
    parser.add_argument('--expression_loss_factor', type=float,
                        help='Center update rate for center loss.', default=0.1)
    parser.add_argument('--loss_weight_base1', type=float,
                        help='The base of the weight of the sub-loss in the full loss.', default=0)
    parser.add_argument('--loss_weight_base2', type=float,
                        help='The base of the weight of the second sub-loss in the full loss.', default=0)
    parser.add_argument('--loss_weight_base3', type=float,
                        help='The base of the weight of the third sub-loss in the full loss.', default=0)
    parser.add_argument('--downsample', type=int,
                        help='The base of the weight of the third sub-loss in the full loss.', default=20)

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--trainset_start', type=int,
        help='Number of the start of the train set', default=0)
    parser.add_argument('--trainset_end', type=int,
        help='Number of the end of the train set')
    parser.add_argument('--evaluate_mode', type=str,
                        help='The evaluation mode: Euclidian distance or similarity by cosine distance.',
                        default='Euclidian')
    parser.add_argument('--far', type=float,
                        help='FAR/ false acception rate (false positive rate) for evaluating the validation / recall rate',
                        default=0.01)


    parser.add_argument('--expr_pairs', type=str,
                        help='Path to the data directory containing the aligned face with expressions for face verification validation.', default='../data/IdentitySplit_4th_10fold_oulucasiapairs_Six.txt')
    parser.add_argument('--train_pairs', type=str,
                        help='Path to the file containing the training image-pairs for training.',
                        default='../data/IdentitySplit_4th_10fold_oulucasiapairs_Six.txt')
    parser.add_argument('--test_pairs', type=str,
                        help='Path to the file containing the training image-pairs for test.',
                        default='../data/IdentitySplit_4th_10fold_oulucasiapairs_Six.txt')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
