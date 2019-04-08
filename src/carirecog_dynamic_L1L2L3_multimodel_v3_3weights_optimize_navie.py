

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


#### libs of DavaideSanderburg ####
sys.path.insert(0, '../lib/facenet/src')
import facenet
import lfw

###### user custom lib
import facenet_ext
import lfw_ext
import metrics_loss
import train_BP

### FER2013 ###
#EXPRSSIONS_TYPE =  ['0=Angry', '1=Disgust', '2=Fear', '3=Happy', '4=Sad', '5=Surprise', '6=Neutral']
# ### CK+  ###
#EXPRSSIONS_TYPE =  ['0=neutral', '1=anger', '2=contempt', '3=disgust', '4=fear', '5=happy', '6=sadness', '7=surprise']
## OULU-CASIA  ###
EXPRSSIONS_TYPE =  ['0=Anger', '1=Disgust', '2=Fear', '3=Happiness', '4=Sadness', '5=Surprise' ]
###EXPRSSIONS_TYPE =  ['0=Neutral', '1=Anger', '2=Disgust', '3=Fear', '4=Happiness', '5=Sadness', '6=Surprise' ]
# ### SFEW  ###
# EXPRSSIONS_TYPE =  ['0=Angry', '1=Disgust', '2=Fear', '3=Happy', '4=Neutral', '5=Sad', '6=Surprise']

def main(args):
    
    module_networks = str.split(args.model_def,'/')[-1]
    network = imp.load_source(module_networks, args.model_def)  
    #etwork = importlib.import_module(args.model_def, 'inference')
    #network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)


    #########################   CARI    ##########################
    ## See_test samples
    # args.test_pairs = '/data/zming/datasets/CaVINet-master/train-test-files/training_zm.txt'
    image_list_test, label_list_test, nrof_classes_test, label_verif, label_cari_test \
        = facenet_ext.get_image_paths_and_labels_cavi_fromfile(args.test_pairs, args.data_dir)

    ## downsampling the test samples
    #down_sample = 4
    #down_sample = 10
    down_sample = args.downsample
    xx = zip(image_list_test[0::2], image_list_test[1::2], label_list_test[0::2], label_list_test[1::2],
             label_cari_test[0::2], label_cari_test[1::2])
    yy = xx[0::down_sample]
    # image_list_test = zip(*yy)[0] + zip(*yy)[1]
    list_img_tmp = []
    list_label_tmp = []
    list_cari_tmp = []
    for y in yy:
        list_img_tmp.append(y[0])
        list_img_tmp.append(y[1])
        list_label_tmp.append(y[2])
        list_label_tmp.append(y[3])
        list_cari_tmp.append(y[4])
        list_cari_tmp.append(y[5])
    image_list_test = list_img_tmp
    label_list_test = list_label_tmp
    label_cari_test = list_cari_tmp
    label_verif = label_verif[0::down_sample]


    ## removing the repeat images in the test dataset for identification test
    xx = zip(image_list_test, label_list_test, label_cari_test)
    yy = list(set(xx))
    zz = zip(*yy)
    image_list_test_id = zz[0]
    label_list_id_test = zz[1]
    label_cari_test = zz[2]
    list_class_test = list(set(label_list_id_test))
    list_class_test = [x.upper() for x in list_class_test]

    ### all samples
    image_list_all, label_list_id_all, label_list_cari_all, nrof_classes_all = facenet_ext.get_image_paths_and_labels_cavi(args.data_dir)

    ## remove the test images
    list_train_idx = []
    for i, img in enumerate(image_list_all):
        if not img in image_list_test_id:
            list_train_idx += [i]
    shuffle(list_train_idx)
    image_list = (np.array(image_list_all)[list_train_idx]).tolist()
    label_list_id = (np.array(label_list_id_all)[list_train_idx]).tolist()
    label_list_cari = (np.array(label_list_cari_all)[list_train_idx]).tolist()
    label_list = label_list_id
    nrof_classes = len(list(set(label_list)))

    ## select the id as same as the test images: 195 classies
    list_train_idx = []
    for i, label in enumerate(label_list_id):
        if label.upper() in list_class_test:
            list_train_idx += [i]
    image_list = (np.array(image_list)[list_train_idx]).tolist()
    label_list_id = (np.array(label_list_id)[list_train_idx]).tolist()
    label_list_cari = (np.array(label_list_cari)[list_train_idx]).tolist()
    label_list = label_list_id
    nrof_classes = len(list(set(label_list)))

    # ## training samples, the samples saving in the training or test file is the pair of images
    # image_list, label_list_id, nrof_classes, _, label_list_cari\
    #     = facenet_ext.get_image_paths_and_labels_cavi_fromfile(args.train_pairs, args.data_dir)
    # ## removing the repeat images of the reading image pairs
    # xx = zip(image_list, label_list_id, label_list_cari)
    # yy = list(set(xx))
    # zz = zip(*yy)
    # image_list = zz[0]
    # label_list_id = zz[1]
    # label_list_cari = zz[2]
    # nrof_classes = len(set(label_list_id))

    # #### t-sne ########
    # image_list_test = image_list
    # label_list_test = label_list_id
    # image_list_test_id = image_list
    # label_list_id_test = label_list_id
    # label_cari_test = label_list_cari
    # label_verif = np.tile([0,1],int(len(image_list)/4)).tolist()
    # #### t-sne ############

    ## mapping the string id label to the number id label
    see_id = list(set(label_list_id))
    see_id.sort()
    See_id = []
    for id in see_id:
        See_id.append(id.upper())

    label_tmp = []
    for label in label_list_id:
        label_tmp.append(See_id.index(label.upper()))
    label_list_id = label_tmp
    label_list = label_list_id

    label_tmp = []
    for label in label_list_id_test:
        label_tmp.append(See_id.index(label.upper()))
    label_list_id_test = label_tmp

    label_tmp = []
    for label in label_list_test:
        label_tmp.append(See_id.index(label.upper()))
    label_list_test = label_tmp
    ####################################################################################
    ## filtering the visual images in test dataset
    filter = [x==0 for x in label_cari_test]
    image_list_test_id_visual = list(compress(image_list_test_id, filter))
    label_list_id_test_visual = list(compress(label_list_id_test, filter))

    # filtering the caricature images in test dataset
    filter = [x==1 for x in label_cari_test]
    image_list_test_id_cari = list(compress(image_list_test_id, filter))
    label_list_id_test_cari = list(compress(label_list_id_test, filter))
    #filter = [x == 1 for x in label_list_cari]
    # image_list_test_id_cari = list(compress(image_list, filter))
    # label_list_id_test_cari = list(compress(label_list_id, filter))

    print('Total number of subjects: %d' % nrof_classes)
    print('Total number of images: %d' % len(image_list))

    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.pretrained_model))
        print('Pre-trained model: %s' % pretrained_model)
    
    # if args.lfw_dir:
    #     print('LFW directory: %s' % args.lfw_dir)
    #     # Read the file containing the pairs used for testing
    #     pairs = lfw_ext.read_pairs(os.path.expanduser(args.lfw_pairs))
    #     # Get the paths for the corresponding images
    #     lfw_paths, actual_issame = lfw_ext.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

    # if args.expr_pairs:
    #     print('Expression validation dataset directory: %s' % args.expr_pairs)
    #     # Read the file containing the pairs used for testing
    #     expr_pairs = lfw_ext.read_pairs(os.path.expanduser(args.expr_pairs))
    #     # Get the paths for the corresponding images
    #     expr_pair_paths, expr_pair_id_actual_issame = lfw_ext.get_expr_paths(expr_pairs)

    if args.evaluate_express:
        print('Test data directory: %s' % args.data_dir)

        tf.set_random_seed(args.seed)
        ## the global_step is saved as no name variable in the pretrained model, so adding the name 'global_step' will failed to load
        #global_step = tf.Variable(0, trainable=False, name='global_step')
        global_step = tf.Variable(0, trainable=False)


        
        # Create a queue that produces indices into the image_list and label_list 
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
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
        #nrof_preprocess_threads = 1
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label_id, label_expr, label_cari = input_queue.dequeue()
           # filenames, label_id, label_expr = input_queue.dequeue_up_to()
            images = []
            #for filename in tf.unpack(filenames): ## tf0.12
            for filename in tf.unstack(filenames): ## tf1.0
                file_contents = tf.read_file(filename)
                image = tf.image.decode_png(file_contents)
                #image = tf.image.decode_jpeg(file_contents)
                if args.random_rotate:
                    image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    #image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
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
        logits_id = slim.fully_connected(prelogits, nrof_classes, activation_fn=None, weights_initializer= tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits_verif', reuse=False)


	
        # Add center loss
        if args.center_loss_factor>0.0:
            prelogits_center_loss_verif, prelogits_center_loss_verif_n, centers, _, centers_cts_batch_reshape, diff_mean \
                = metrics_loss.center_loss(embeddings, label_id_batch, args.center_loss_alfa, nrof_classes)
            #prelogits_center_loss, _ = facenet.center_loss_similarity(prelogits, label_batch, args.center_loss_alfa, nrof_classes) ####Similarity cosine distance, center loss
            #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss_verif * args.center_loss_factor)

        cross_entropy_verif = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_id, labels=label_id_batch, name='cross_entropy_batch_verif')
        cross_entropy_mean_verif = tf.reduce_mean(cross_entropy_verif, name='cross_entropy_verif')

        loss_verif_n = cross_entropy_verif + args.center_loss_factor*prelogits_center_loss_verif_n
        #loss_verif_n = cross_entropy_verif
        loss_verif = tf.reduce_mean(loss_verif_n, name='loss_verif')
        #loss_verif = tf.add_n([loss_verif_n], name='loss_verif')
        #tf.add_to_collection('losses', cross_entropy_mean_verif)

        ###########################     Branch Visual image recognition  ################################
        # ### transfer the output of the prelogits to 1600 elements which can respahe to 40x40 as an image input of the
        # ### expression net
        # logits_deconv = slim.fully_connected(prelogits, 1600, activation_fn=None,
        #         weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #         weights_regularizer=slim.l2_regularizer(args.weight_decay),
        #         scope='logits_deconv', reuse=False)
        # ### reshape the prelogits to the 2D images [batch, width, height]
        # prelogits_deconv = tf.reshape(logits_deconv, [batch_size_placeholder, 40,40,1], name='prelogits_reshape')

        ### the expression net for expression classification, and the input is the reshaped logits_deconv
        #inputs = end_points['MaxPool_3a_3x3']
        #inputs = end_points['Conv2d_2b_3x3']
        #inputs = end_points['Mixed_6a']
        #inputs = end_points['Mixed_5a']
        #inputs = end_points['Conv2d_4b_3x3']
        #inputs = image_batch
        inputs = end_points['Mixed_7a']
        #inputs = end_points['Mixed_8a']
        #inputs = end_points['Mixed_6a']
        #inputs = end_points['Mixed_6.5a']
        prelogits_expression, end_points_expression = network.inference_expression(inputs, keep_probability_placeholder, phase_train=phase_train_placeholder_expression, weight_decay=args.weight_decay)
        embeddings_expression = tf.nn.l2_normalize(prelogits_expression, 1, 1e-10, name='embeddings_expression')

        mask_expr = tf.equal(label_cari_batch, 0)
        embeddings_expression_filter = tf.boolean_mask(embeddings_expression, mask_expr)
        label_id_filter_expr = tf.boolean_mask(label_id_batch, mask_expr)

        prelogits_expression_center_loss, prelogits_expression_center_loss_n, centers_expression, _, centers_cts_batch_reshape_expression, diff_mean_expression \
            = metrics_loss.center_loss(embeddings_expression_filter, label_id_filter_expr, args.center_loss_alfa, nrof_classes)

        #logits_0 = slim.fully_connected(prelogits_expression, 128, activation_fn=tf.nn.relu,weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits_0', reuse=False)

        #logits_0 = slim.dropout(logits_0, keep_probability_placeholder, is_training=True, scope='Dropout')

        #logits_expr = slim.fully_connected(logits_0, len(set(label_list)), activation_fn=tf.nn.relu,weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits', reuse=False)
        logits_expr = slim.fully_connected(prelogits_expression, len(set(label_list)), activation_fn=tf.nn.relu,weights_initializer=tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits', reuse=False)

        logits_expr = tf.identity(logits_expr, 'logits_expr')

        ## Filtering the visual image for training the Branch Visual recognition
        
        logits_expr_filter = tf.boolean_mask(logits_expr, mask_expr)
        


        # Calculate the average cross entropy loss across the batch
        cross_entropy_expr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_expr_filter, labels=label_id_filter_expr, name='cross_entropy_per_example')
        cross_entropy_mean_expr = tf.reduce_mean(cross_entropy_expr, name='cross_entropy_expr')
        #tf.add_to_collection('losses', cross_entropy_mean)
        
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # loss_expr_n = cross_entropy_expr
        loss_expr_n = cross_entropy_expr+args.center_loss_factor*prelogits_expression_center_loss_n
        loss_expr = tf.reduce_mean(loss_expr_n, name='loss_expr')
        #loss_expr = tf.add_n([cross_entropy_mean_expr]+[args.center_loss_factor*prelogits_expression_center_loss], name='loss_expr')
        #loss_expr = tf.add_n([loss_expr_n], name='loss_expr')


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
        logits_cari = slim.fully_connected(prelogits_cari, len(set(label_list_id)), activation_fn=tf.nn.relu,
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
        #loss_full = tf.multiply(tf.transpose(softmax_lossweights),
        #loss_full = tf.add_n([loss_verif]+[loss_expr_percentage*loss_expr], name='loss_full')
        #loss_full = tf.add_n([1*loss_verif]+[args.expression_loss_factor*loss_expr], name='loss_full')
        #loss_full = tf.matmul(tf.transpose(softmax_lossweights), [loss_verif, loss_expr])


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
        update_gradient_vars_mainstem = []
        update_gradient_vars_weights = []

        # for var in tf.trainable_variables():
        #     #Update variables for Branch Expression recogntion
        #     if 'InceptionResnetV1_expression/' in var.op.name or 'Logits/' in var.op.name or 'Logits_0/' in var.op.name:
        #             print(var.op.name)
        #             update_gradient_vars_expr.append(var)
        #     # Update variables for Branch face verification
        #     elif 'InceptionResnetV1/Block8' in var.op.name or 'InceptionResnetV1/Repeat_2/block8' in var.op.name or 'Logits_verif/' in var.op.name:
        #         print(var.op.name)
        #         update_gradient_vars_verif.append(var)
        #
        #     # Update variables for main stem
        #     else:
        #         print(var.op.name)
        #         update_gradient_vars_mainstem.append(var)
        #
        #     #update_gradient_vars_mainstem.append(var)

        #update_gradient_vars_mainstem = tf.trainable_variables()
        # for var in tf.trainable_variables():
        #     # Update variables for dynmic weights
        #     if 'Logits_lossweights/' not in var.op.name:
        #         update_gradient_vars_mainstem.append(var)

        # for var in tf.trainable_variables():
        #     # Update variables for dynmic weights
        #     if 'Logits_lossweights/' in var.op.name or 'Layer_lossweights' in var.op.name:
        #         update_gradient_vars_weights.append(var)
        #     else:
        #         update_gradient_vars_mainstem.append(var)

        update_gradient_vars_mainstem = tf.trainable_variables()

        paracnt, parasize = count_paras(update_gradient_vars_verif)
        print('The number of the updating parameters in the model Facenet is %dM, ......the size is : %dM bytes'%(paracnt/1e6, parasize/1e6))

        paracnt, parasize = count_paras(update_gradient_vars_expr)
        print('The number of the update parameters in the model Facial Expression is %dM, ......the size is : %dM bytes'%(paracnt/1e6, parasize/1e6))

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # train_op_verif, grads_verif, grads_clip_verif = train_BP.train(loss_verif, global_step, args.optimizer,
        #     learning_rate, args.moving_average_decay, update_gradient_vars_verif, summary_op, args.log_histograms)
        # train_op_expr, grads_expr, grads_clip_expr = train_BP.train(loss_expr, global_step, args.optimizer,
        #     learning_rate, args.moving_average_decay, update_gradient_vars_expr, summary_op, args.log_histograms)
        train_op_mainstem, grads_full, grads_clip_full = train_BP.train(loss_full, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, update_gradient_vars_mainstem, summary_op, args.log_histograms)
        # train_op_weights, grads_weights, grads_clip__weights = train_BP.train(loss_full, global_step, args.optimizer,
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
                    # for var in tf.global_variables():
                    #     if 'center' not in var.op.name:
                    #         restore_vars.append(var)
                    #
                    # paracnt, parasize = count_paras(restore_vars)
                    # print('The number of the loading parameters in the model(FaceLiveNet) is %dM, ......the size is : %dM bytes' % (
                    #         paracnt / 1e6, parasize / 1e6))
                    # restore_saver_expression = tf.train.Saver(restore_vars)
                    # restore_saver_expression.restore(sess,os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))

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

            nrof_expressions = len(set(label_list))
            each_expr_acc = np.zeros(nrof_expressions)
            express_probs_confus_matrix = np.zeros((nrof_expressions,nrof_expressions))
            # with open(os.path.join(log_dir, 'Authentication_result.txt'), 'at') as f:
            #     f.write('step, acc_faceauthen, best_acc_faceauthen, acc_expression, acc_verif_exprpairs\n')
            # with open(os.path.join(log_dir, 'LFW_result.txt'), 'at') as f:
            #     f.write('step, acc, val, far, best_acc\n')
            # with open(os.path.join(log_dir, 'Expr_paris_result.txt'), 'at') as f:
            #     f.write('step, acc, val, far, best_acc\n')
            # with open(os.path.join(log_dir, 'Expression_result.txt'), 'at') as f:
            #     f.write('step, test_expr_acc, best_acc_exprecog\n')

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
                    = train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op,
                            image_paths_placeholder, labels_id_placeholder, labels_expr_placeholder,
                            learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                            loss_verif, loss_expr, summary_op, summary_writer,
                            regularization_losses, args.learning_rate_schedule_file, prelogits_center_loss_verif,
                            cross_entropy_mean_verif, cross_entropy_mean_expr, acc, val, far, centers_cts_batch_reshape,
                            logits_id, logits_expr, keep_probability_placeholder, update_gradient_vars_expr, acc_expression,
                            each_expr_acc, label_batch_id, label_batch_expr, express_probs_confus_matrix, log_dir,
                            model_dir, image_batch, learning_rate, phase_train_placeholder_expression,
                            best_acc_exprecog, label_list_id, softmax_acc_verif, softmax_acc_expr, cross_entropy_verif, diff_mean,
                            centers, acc_expr_paris, val_expr_paris, far_expr_paris, best_acc_faceverif_expr,
                            best_acc_faceverif_lfw, train_op_mainstem, best_acc_faceauthen, best_authen_verif_exprpairs,
                            best_authen_exprecog, loss_verif_percentage, loss_expr_percentage, epoch_current, logits_lossweights_embedings,
                            label_list_cari, labels_cari_placeholder, phase_train_placeholder_cari, softmax_acc_cari,
                            loss_cari, cross_entropy_mean_cari, loss_cari_percentage, acc_cari, best_acc_carirecog,
                            acc_v2c, acc_c2v, best_acc_c2v, best_acc_v2c, acc_mixrecog, best_acc_mixrecog,
                            loss_for_weights, f_weights, f_loss)


                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                ## Evaluate on LFW
                # if(epoch%20==0):
                #      if args.lfw_dir:
                #          acc, val, far = evaluate(sess, enqueue_op, image_paths_placeholder, labels_id_placeholder,
                #                                   labels_expr_placeholder, phase_train_placeholder, batch_size_placeholder,embeddings, label_id_batch,
                #                                   lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds,
                #                                   log_dir, step, summary_writer, args.evaluate_mode,
                #                                   keep_probability_placeholder, 'LFW', best_acc_faceverif_lfw, args)
                if (epoch % 1 == 0):
                     if args.expr_pairs:
                         acc_expr_paris, val_expr_paris, far_expr_paris, acc_mixrecog = evaluate(sess, enqueue_op, image_paths_placeholder, labels_id_placeholder,
                                                  labels_expr_placeholder, phase_train_placeholder, batch_size_placeholder,embeddings, label_id_batch,
                                                  image_list_test, label_verif, args.lfw_batch_size, args.lfw_nrof_folds,
                                                  log_dir, step, summary_writer, args.evaluate_mode,
                                                  keep_probability_placeholder, 'cari-verif-pairs', best_acc_faceverif_expr,
                                                  args, logits_id, label_list_test)

                ## Evaluate for visual image classification
                if not (epoch % 1 ):
                    if args.evaluate_express:
                        acc_expression, each_expr_acc, exp_cnt, expredict_cnt,  express_probs_confus_matrix, express_recog_images\
                            = evaluate_expression(sess, batch_size_placeholder, logits_expr,
                                                  image_list_test_id_visual, label_list_id_test_visual, 100,
                                                  log_dir, step, summary_writer, keep_probability_placeholder,input_queue,
                                                  phase_train_placeholder_expression, phase_train_placeholder, args,
                                                  best_acc_exprecog, phase_train_placeholder_cari, 'visual', embeddings)

                ## Evaluate for caricature image classification
                if not (epoch % 1):
                    acc_cari, each_cari_acc, cari_cnt, caripredict_cnt, cari_probs_confus_matrix, cari_recog_images \
                        = evaluate_expression(sess, batch_size_placeholder, logits_cari,
                                              image_list_test_id_cari, label_list_id_test_cari, 100,
                                              log_dir, step, summary_writer, keep_probability_placeholder, input_queue,
                                              phase_train_placeholder_expression, phase_train_placeholder, args,
                                              best_acc_carirecog, phase_train_placeholder_cari, 'cari', embeddings)

                ## Evaluate for visual to caricature image classification
                if not (epoch % 1):
                    acc_v2c, each_v2c_acc, v2c_cnt, caripredict_v2c, v2c_probs_confus_matrix, v2c_recog_images \
                        = evaluate_expression(sess, batch_size_placeholder, logits_cari,
                                              image_list_test_id_visual, label_list_id_test_visual, 100,
                                              log_dir, step, summary_writer, keep_probability_placeholder, input_queue,
                                              phase_train_placeholder_expression, phase_train_placeholder, args,
                                              best_acc_v2c, phase_train_placeholder_cari, 'visual2cari', embeddings)

                ## Evaluate for caricature to visual image classification
                if not (epoch % 1 ):
                    acc_c2v, each_c2v_acc, c2v_cnt, c2vpredict_cnt,  c2v_probs_confus_matrix, c2v_recog_images\
                        = evaluate_expression(sess, batch_size_placeholder, logits_expr,
                                              image_list_test_id_cari, label_list_id_test_cari, 100,
                                              log_dir, step, summary_writer, keep_probability_placeholder,input_queue,
                                              phase_train_placeholder_expression, phase_train_placeholder, args,
                                              best_acc_c2v, phase_train_placeholder_cari, 'cari2visual', embeddings)



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

                # ## saving the best_model for face verfication on LFW
                # if acc > best_acc_faceverif_lfw:
                #     best_acc_faceverif_lfw = acc
                #     best_model_dir = os.path.join(model_dir, 'best_model_veriflfw')
                #     if not os.path.isdir(best_model_dir):  # Create the log directory if it doesn't exist
                #         os.makedirs(best_model_dir)
                #     if os.listdir(best_model_dir):
                #         for file in glob.glob(os.path.join(best_model_dir, '*.*')):
                #             os.remove(file)
                #     for file in glob.glob(os.path.join(model_dir, '*.*')):
                #         shutil.copy(file, best_model_dir)

                ## saving the best_model for visual recognition
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

                    # ######################## SAVING BEST CONFUSION RESULTS IMAGES  ################################
                    # images_exists = glob.glob(os.path.join(best_model_dir, 'confusion_matrix_images*'))
                    # for folder in images_exists:
                    #     shutil.rmtree(folder)
                    #
                    # confus_images_folder = os.path.join(best_model_dir, 'confusion_matrix_images_%dsteps' % step)
                    #
                    # os.mkdir(confus_images_folder)
                    #
                    # ## mkdir for the confusion matrix of each expression
                    # for i in range(nrof_expressions):
                    #     gt_folder = os.path.join(confus_images_folder, '%d') % i
                    #     os.mkdir(gt_folder)
                    #     for j in range(nrof_expressions):
                    #         predict_folder = os.path.join(confus_images_folder, '%d', '%d') % (i, j)
                    #         os.mkdir(predict_folder)
                    #
                    # ## copy the predicting images to the corresponding folder of the predicting expression
                    # for i, labs_predict in enumerate(express_recog_images):
                    #     for j, lab_predict in enumerate(labs_predict):
                    #         dst = os.path.join(confus_images_folder, '%d', '%d') % (i, j)
                    #         for img in express_recog_images[i][j]:
                    #             shutil.copy(img[0], dst)
                    # ######################## SAVING BEST CONFUSION RESULTS IMAGES  ################################

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

                # ###################   Saving the confusion matrix  ##########################################
                # with open(os.path.join(log_dir, 'confusion_matrix.txt'), 'a') as f:
                #     f.write('%d expressions recognition TRAINING accuracy is: %2.4f\n' % (nrof_expressions, softmax_acc_expr_))
                #     f.write('loss_verif: %2.4f  loss_expr: %2.4f  crossentropy: %2.4f  regloss: %2.4f  centerloss: %2.4f  verifAcc: %2.4f  lr:%e \n' % (loss_verif_, loss_expr_, cross_entropy_mean_expr_, Reg_loss,  center_loss_, verifacc, learning_rate_))
                #     line = ''
                #     for idx, expr in enumerate(EXPRSSIONS_TYPE):
                #         line += (expr + ': %2.4f,  ')%train_each_expr_acc[idx]
                #     f.write('Training acc: '+line + '\n')
                #
                #     f.write('%d expressions recognition TEST accuracy is: %2.4f\n' % (nrof_expressions, acc_expression))
                #     #f.write('>>>>>>>>>>>>>>>>>>>>>>>>>>> Gradient norm**2 is: %f\n' % grads_total_sum)
                #     f.write('--------  Confusion matrix expressions recog AFTER %d steps of the iteration: ---------------\n' % step)
                #     line = ''
                #     for expr in EXPRSSIONS_TYPE:
                #         line += expr + ',  '
                #     f.write(line + '\n')
                #
                #     for i in range(nrof_expressions):
                #         line = ''
                #         line += '%d   ' % i
                #         for j in range(nrof_expressions):
                #             line += '%2.4f ' % express_probs_confus_matrix[i][j]
                #         f.write(line + '\n')
                #     f.write('----------------------------------------------------------------------------------------\n')
                # ###################   Saving the confusion matrix  ##########################################

    return model_dir
  
def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold
  
def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class,trainset_start):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if label >= trainset_start:
                if image in filtered_dataset[label-trainset_start].image_paths:
                    filtered_dataset[label-trainset_start].image_paths.remove(image)
                if len(filtered_dataset[label-trainset_start].image_paths)<min_nrof_images_per_class:
                    removelist.append(label-trainset_start)


        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset
  
def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_id_placeholder, labels_expr_placeholder, learning_rate_placeholder, phase_train_placeholder,
          batch_size_placeholder, global_step, loss_verif, loss_expr, summary_op,
          summary_writer, regularization_losses, learning_rate_schedule_file, prelogits_center_loss_verif,
          cross_entropy_mean_verif, cross_entropy_mean_expr, acc, val, far, centers_cts_batch_reshape, logits_id,
          logits_expr, keep_probability_placeholder, update_gradient_vars_expr, acc_expression, each_expr_acc,
          label_batch_id, label_batch_expr, express_probs_confus_matrix, log_dir, model_dir,
          image_batch, learning_rate, phase_train_placeholder_expression, best_acc_exprecog, label_list_id, softmax_acc_verif,
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
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch_current)

    print('Index_dequeue_op....')
    index_epoch = sess.run(index_dequeue_op)
    label_id_epoch = np.array(label_list_id)[index_epoch]
    label_expr_epoch = np.array(label_list)[index_epoch]
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
            # #err, _, step, reg_loss, prelogits_center_loss_verif_, cross_entropy_mean_, centers_cts_batch_reshape_, softmax_acc_, logits_, summary_str = sess.run([loss, train_op, global_step, regularization_losses, prelogits_center_loss_verif, cross_entropy_mean, centers_cts_batch_reshape, softmax_acc, logits, summary_op], feed_dict=feed_dict)
            # loss_verif_, loss_expr_, _, _, step, reg_loss, prelogits_center_loss_verif_, cross_entropy_mean_verif_, \
            # cross_entropy_mean_expr_, centers_cts_batch_reshape_, logits_id_, logits_expr_, label_batch_id_, \
            # label_batch_expr_, grads_expr_, grads_clip_expr_, image_batch_, learning_rate_, softmax_acc_verif_, \
            # softmax_acc_expr_, cross_entropy_verif_, diff_mean_, centers_, _, loss_verif_percentage_, \
            # loss_expr_percentage_, summary_str \
            #     = sess.run([loss_verif, loss_expr, train_op_verif, train_op_expr, global_step, regularization_losses,
            #                 prelogits_center_loss_verif, cross_entropy_mean_verif, cross_entropy_mean_expr,
            #                 centers_cts_batch_reshape, logits_id, logits_expr, label_batch_id, label_batch_expr, grads_expr,
            #                 grads_clip_expr, image_batch, learning_rate, softmax_acc_verif, softmax_acc_expr,
            #                 cross_entropy_verif, diff_mean, centers, train_op_mainstem, loss_verif_percentage,
            #                 loss_expr_percentage, summary_op], feed_dict=feed_dict)
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
            # #err, _, step, reg_loss, prelogits_center_loss_verif_, cross_entropy_mean_, centers_cts_batch_reshape_, softmax_acc_, logits_ = sess.run([loss, train_op, global_step, regularization_losses,prelogits_center_loss_verif, cross_entropy_mean, centers_cts_batch_reshape, softmax_acc, logits], feed_dict=feed_dict)
            # loss_verif_, loss_expr_, _, _, step, reg_loss, prelogits_center_loss_verif_, cross_entropy_mean_verif_, \
            # cross_entropy_mean_expr_, centers_cts_batch_reshape_, logits_id_, logits_expr_, label_batch_id_, \
            # label_batch_expr_, grads_expr_, grads_clip_expr_, image_batch_, learning_rate_, softmax_acc_verif_, \
            # softmax_acc_expr_, cross_entropy_verif_, diff_mean_, centers_, _, loss_verif_percentage_, \
            # loss_expr_percentage_ \
            #     = sess.run([loss_verif, loss_expr,  train_op_verif, train_op_expr, global_step, regularization_losses,
            #                 prelogits_center_loss_verif, cross_entropy_mean_verif, cross_entropy_mean_expr,
            #                 centers_cts_batch_reshape, logits_id, logits_expr, label_batch_id, label_batch_expr, grads_expr,
            #                 grads_clip_expr, image_batch, learning_rate, softmax_acc_verif, softmax_acc_expr,
            #                 cross_entropy_verif, diff_mean,centers, train_op_mainstem, loss_verif_percentage,
            #                 loss_expr_percentage], feed_dict=feed_dict)
            #err, _, step, reg_loss, prelogits_center_loss_verif_, cross_entropy_mean_, centers_cts_batch_reshape_, softmax_acc_, logits_ = sess.run([loss, train_op, global_step, regularization_losses,prelogits_center_loss_verif, cross_entropy_mean, centers_cts_batch_reshape, softmax_acc, logits], feed_dict=feed_dict)
            loss_verif_, loss_expr_, step, reg_loss, prelogits_center_loss_verif_, cross_entropy_mean_verif_, \
            cross_entropy_mean_expr_, centers_cts_batch_reshape_, logits_id_, logits_expr_, label_batch_id_, \
            label_batch_expr_, image_batch_, learning_rate_, softmax_acc_verif_, \
            softmax_acc_expr_, cross_entropy_verif_, diff_mean_, centers_, _, loss_verif_percentage_, \
            loss_expr_percentage_, logits_lossweights_embedings_, softmax_acc_cari_, loss_cari_, cross_entropy_mean_cari_, \
            loss_cari_percentage_,loss_for_weights_,\
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


        ####### gradients values checking ####################
        grads_sum = 0
        grad_clip_sum = 0
        #######################################################################################################
        ########### grad[0] is the gradient, and the grad[1] is value of the weights or bias ###
        ########### grads_weights[2i] are the weights of layer i, and the grads_weights_[2i+1] are the bias of layers i###
        #######################################################################################################
        # for i, grad in enumerate(grads_weights_):
        #     grad_norm = LA.norm(np.asarray(grad[0]))
        #     # if math.isnan(grad_norm):
        #     #     print(grad)
        #     grads_sum += grad_norm**2
        #     print ('grad_%dth: %f  '%(i,grad_norm), end='')
        # print('\n')
        # for i, grad_clip in enumerate(grads_clip__weights_):
        #     grad_clip_norm = LA.norm(np.asarray(grad_clip))
        #     grad_clip_sum += grad_clip_norm**2
        #     print ('grad_clip_%dth: %f  '%(i,grad_clip_norm), end='')
        # print('\n')
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>> Gradient norm is: %f' % math.sqrt(grads_sum))
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>> Gradient clip norm is: %f' % math.sqrt(grad_clip_sum))



        # ############# the accuracy of each expression  ################
        # express_probs = np.exp(logits_expr_) / np.tile(
        #     np.reshape(np.sum(np.exp(logits_expr_), 1), (logits_expr_.shape[0], 1)), (1, logits_expr_.shape[1]))
        # nrof_expression = len(set(label_list))
        # expressions_predict = np.argmax(express_probs, 1)
        #
        # exp_cnt = np.zeros(nrof_expression)
        # expredict_cnt = np.zeros(nrof_expression)
        # for i in range(label_batch_expr_.shape[0]):
        #     lab = label_batch_expr_[i]
        #     exp_cnt[lab] += 1
        #     if lab == expressions_predict[i]:
        #         expredict_cnt[lab] += 1
        # train_each_expr_acc = expredict_cnt / exp_cnt
        ###############################################
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
        #print('Training each_expression_acc: 0=Angry %2.4f, 1=Disgust %2.4f, 2=Fear %2.4f, 3=Happy %2.4f, 4=Sad %2.4f, 5=Surprise %2.4f, 6=Neutral %2.4f' % (train_each_expr_acc[0], train_each_expr_acc[1], train_each_expr_acc[2], train_each_expr_acc[3], train_each_expr_acc[4], train_each_expr_acc[5],train_each_expr_acc[6]))
        #print ('Test each_expression_acc: 0=Angry %2.4f, 1=Disgust %2.4f, 2=Fear %2.4f, 3=Happy %2.4f, 4=Sad %2.4f, 5=Surprise %2.4f, 6=Neutral %2.4f'%(each_expr_acc[0], each_expr_acc[1], each_expr_acc[2], each_expr_acc[3], each_expr_acc[4], each_expr_acc[5], each_expr_acc[6]))
        #print('Face Authentication : acc_auth %f, best_acc_authen %f (acc_expr_pairs:%f, acc_exprrecog: %f)'%(acc_expr_paris*acc_expression, best_acc_faceauthen, best_authen_verif_exprpairs, best_authen_exprecog))
        #print('Face verification on LFW: acc_LFW %f, val_LFW %f, far_LFW %f, best_acc_faceverif_LFW %f'%(acc, val, far, best_acc_faceverif_lfw))
        print('Verification on Cari-Visual pairs: acc %f, val %f, far %f, best_acc %f'%(acc_expr_paris, val_expr_paris, far_expr_paris, best_acc_faceverif_expr))
        print('Visual image recognition : test_visual_acc %2.4f, best_visual_acc %2.4f'%(acc_expression, best_acc_exprecog))
        print('Caricature image recognition : test_cari_acc %2.4f, best_cari_acc %2.4f'%(acc_cari, best_acc_carirecog))
        print('C2V image recognition : test_c2v_acc %2.4f, best_c2v_acc %2.4f'%(acc_c2v, best_acc_c2v))
        print('V2C image recognition : test_v2c_acc %2.4f, best_v2c_acc %2.4f'%(acc_v2c, best_acc_v2c))
        print('Mix image recognition : test_mix_acc %2.4f, best_mix_acc %2.4f' % (acc_mixrecog, best_acc_mixrecog))

        # if np.exp(np.mean(logits_lossweights_embedings_[:,0]))==1.0 or np.exp(np.mean(logits_lossweights_embedings_[:,1])) == 1.0 or np.exp(np.mean(logits_lossweights_embedings_[:,2])) == 1.0:
        #     print ('stop')


        # print('Training each_expression_acc: ',end = '')
        # for i, expr in enumerate(EXPRSSIONS_TYPE):
        #     print(expr+'  %2.4f  '%(train_each_expr_acc[i]),end='')
        # print('\n')
        #
        # print('Test each_expression_acc: ',end = '')
        # for i, expr in enumerate(EXPRSSIONS_TYPE):
        #     print(expr+'  %2.4f  '%(each_expr_acc[i]),end='')
        # print('\n')
       
        # print('---------------------- Confusion matrix expressions recog ----------------------\n')
        # for expr in EXPRSSIONS_TYPE:
        #     print(expr+',  ',end='')
        # print('\n')
        #
        # for i in range(nrof_expression):
        #     print ('%d   '%i, end='')
        #     for j in range(nrof_expression):
        #         print ('%2.4f '%express_probs_confus_matrix[i][j], end='')
        #     print('\n')
        # print('----------------------------------------------------------------------------------------\n')


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
        _, _, accuracy, val, val_std, far, fp_idx, fn_idx,best_threshold, val_threshold = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    if evaluate_mode == 'similarity':
        pca = PCA(n_components=128)
        pca.fit(emb_array)
        emb_array_pca = pca.transform(emb_array)
        _, _, accuracy, val, val_std, far, fp_idx, fn_idx,best_threshold, val_threshold = lfw.evaluate_cosine(emb_array_pca, actual_issame, nrof_folds=nrof_folds)


    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag=dataset+'/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag=dataset+'/val_rate', simple_value=val)
    summary.value.add(tag=dataset + '/far_rate', simple_value=far)
    summary.value.add(tag='time/'+dataset, simple_value=lfw_time)
    summary_writer.add_summary(summary, step)

    acc = np.mean(accuracy)
    if acc > best_acc:
        np.save(os.path.join(log_dir, 'features_cari-visual-verif_emb.npy'), emb_array)
        np.save(os.path.join(log_dir, 'features_cari-visual-verif_label.npy'), id_actual)
        np.save(os.path.join(log_dir, 'features_cari-visual-verif_pairlabel.npy'), actual_issame)

    with open(os.path.join(log_dir,dataset+'_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\t%.5f\t%.5f\t%f\n' % (step, acc, val, far, best_acc, test_id_acc))



    return acc, val, far, test_id_acc


def evaluate_expression(sess,
             batch_size_placeholder,
             logits, image_paths, actual_expre, batch_size, log_dir, step, summary_writer,
             keep_probability_placeholder,input_queue,phase_train_placeholder_expression, phase_train_placeholder, args,
             best_acc_recog, phase_train_placeholder_cari, imgtype, embeddings):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    #print('Runnning forward pass on FER2013 images')
    print('Runnning forward pass on expression images')
    nrof_images = len(actual_expre)

    #batch_size = 128

    ############## Enqueue complete batches ##############################
    #nrof_batches = nrof_images // batch_size ## The floor division to get the maximum number of the complete batch
    nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))

    #nrof_enqueue = nrof_batches * batch_size
    nrof_enqueue = nrof_images

    ############## Allow enqueue incomplete batch  ##############################
    # nrof_batches = int(math.ceil(nrof_images / batch_size)) ## To get the left elements in the queue when the nrof_images can be not exact divided by batch_size
    # nrof_enqueue = nrof_images

    # Enqueue one epoch of image paths and labels
    #labels_array = np.expand_dims(actual_expre[0:nrof_enqueue], 1)
    labels_array = np.expand_dims(np.arange(nrof_enqueue),1)  ## labels_array is not the label of expression of the image, it is the number of the image in the queue
    image_paths_array = np.expand_dims(np.array(image_paths[0:nrof_enqueue]), 1)
    # sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_id_placeholder: labels_array, labels_expr_placeholder: labels_array})
    #filenames, label = sess.run(input_queue.dequeue())
    logits_size = logits.get_shape()[1]
    embedding_size = embeddings.get_shape()[1]

    logits_array = np.zeros((nrof_enqueue, logits_size), dtype=float)
    emb_array = np.zeros((nrof_enqueue, embedding_size), dtype=float)
    ## label_batch_array is not the label of expression of the image , it is the number of the image in the queue.
    ## label_batch_array is used for keeping the order of the labels and images after the batch_join operation which
    ## generates the batch in multi-thread scrambling the order
    label_batch_array = np.zeros(nrof_enqueue, dtype=int)

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

    for ii in range(nrof_batches):
        print('nrof_batches %d'%ii)
        start_index = ii* batch_size
        end_index = min((ii + 1) * batch_size, nrof_images)
        paths_batch = image_paths[start_index:end_index]
        #### load image including the image whiten operation
        images = facenet_ext.load_data(paths_batch, False, False, args.image_size)
        feed_dict = {phase_train_placeholder: False, phase_train_placeholder_expression: False,
                     phase_train_placeholder_cari: False, batch_size_placeholder: batch_size,
                     keep_probability_placeholder: 1.0, images_placeholder: images}
        ### Capture the exceptions when the queue is exhausted for producing the batch
        try:
            logits_batch,  emb = sess.run([logits, embeddings], feed_dict=feed_dict)
        except tf.errors.OutOfRangeError:
            print('Exceptions: the queue is exhausted !')

        # label_batch_array[lab] = lab
        # logits_array[lab] = logits_batch
        logits_array[start_index:end_index, :] = logits_batch
        emb_array[start_index:end_index, :] = emb
    #assert np.array_equal(label_batch_array, np.arange(nrof_enqueue)) == True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'

    actual_expre_batch = actual_expre[0:nrof_enqueue]
    express_probs = np.exp(logits_array) / np.tile(np.reshape(np.sum(np.exp(logits_array), 1), (logits_array.shape[0], 1)), (1, logits_array.shape[1]))
    nrof_expression = express_probs.shape[1]
    expressions_predict = np.argmax(express_probs, 1)
    #### Training accuracy of softmax: check the underfitting or overfiting #############################
    correct_prediction = np.equal(expressions_predict, actual_expre_batch)
    test_recog_acc = np.mean(correct_prediction)

    ############# the accuracy of each expression  ################
    ### Initializing the confusion matrix
    exp_cnt = np.zeros(nrof_expression)
    expredict_cnt = np.zeros(nrof_expression)
    express_probs_confus_matrix = np.zeros((nrof_expression, nrof_expression))
    express_recog_images = []
    for i in range(nrof_expression):
        express_recog_images.append([])
        for _ in range(nrof_expression):
            express_recog_images[i].append([])

    ### Fill the confusion matrix
    for i in range(label_batch_array.shape[0]):
        lab  = actual_expre_batch[i]
        exp_cnt[lab] += 1
        express_probs_confus_matrix[lab, expressions_predict[i]] += 1
        express_recog_images[lab][expressions_predict[i]].append(image_paths_array[i])
        if  lab == expressions_predict[i]:
            expredict_cnt[lab] += 1
    test_each_expr_acc = expredict_cnt/exp_cnt
    express_probs_confus_matrix /= np.expand_dims(exp_cnt,1)
    ###############################################

    print('%d %s recognition accuracy is: %f' % (nrof_expression, imgtype,test_recog_acc))

    ############### Saving recognition CONFUSION Results images of the 7 expressions  #####################
    # print('Saving expression recognition images corresponding to the confusion matrix in %s...'%log_dir)
    #
    # images_exists = glob.glob(os.path.join(log_dir, 'confusion_matrix_images*'))
    # for folder in images_exists:
    #     shutil.rmtree(folder)
    #
    # confus_images_folder = os.path.join(log_dir, 'confusion_matrix_images_%dsteps' % step)
    #
    # os.mkdir(confus_images_folder)
    #
    # ## mkdir for the confusion matrix of each expression
    # for i in range(nrof_expression):
    #     gt_folder = os.path.join(confus_images_folder, '%d')%i
    #     os.mkdir(gt_folder)
    #     for j in range(nrof_expression):
    #         predict_folder = os.path.join(confus_images_folder, '%d', '%d')%(i,j)
    #         os.mkdir(predict_folder)
    #
    # ## copy the predicting images to the corresponding folder of the predicting expression
    # for i, labs_predict in enumerate(express_recog_images):
    #     for j, lab_predict in enumerate(labs_predict):
    #         dst = os.path.join(confus_images_folder, '%d', '%d')%(i,j)
    #         for img in express_recog_images[i][j]:
    #             shutil.copy(img[0], dst)
    ############### Saving recognition results images of the 7 expressions  #####################

    fer_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='fer/accuracy', simple_value=test_recog_acc)
    summary.value.add(tag='time/fer', simple_value=fer_time)
    summary_writer.add_summary(summary, step)
    #with open(os.path.join(log_dir, 'Fer2013_result.txt'), 'at') as f:
    with open(os.path.join(log_dir, '%s.txt'%imgtype), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, test_recog_acc, best_acc_recog))

    if test_recog_acc>best_acc_recog:
        id_actual = actual_expre_batch
        np.save(os.path.join(log_dir, 'features_%s.npy'%imgtype), emb_array)
        np.save(os.path.join(log_dir, 'features_%s_label.npy'%imgtype), id_actual)


    return test_recog_acc, test_each_expr_acc, exp_cnt, expredict_cnt, express_probs_confus_matrix, express_recog_images
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
