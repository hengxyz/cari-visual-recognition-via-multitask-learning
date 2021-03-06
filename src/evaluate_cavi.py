"""Functions for the evaluation of the dynamic multi-task caricature-visual image face recogntion on dataset cavi.
"""
# MIT License
#
# Copyright (c) 2019 Zuheng Ming

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import math
from sklearn.decomposition import PCA
import time
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import islice
import shutil

sys.path.append('./util')
import  save_false_images
import facenet_ext
import lfw_ext
from itertools import compress
import random


def main(args):

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    print('log_dir: %s\n' % log_dir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:


            #########################   CARI    ##########################
            ## See_test samples
            # args.test_pairs = '/data/zming/datasets/CaVINet-master/train-test-files/training_zm.txt'
            image_list_test, label_list_test, nrof_classes_test, label_verif, label_cari_test \
                = facenet_ext.get_image_paths_and_labels_cavi_fromfile(args.test_pairs, args.data_dir)

            ## downsampling the test samples
            down_sample = 20
            # down_sample = 10
            #down_sample = args.downsample
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

            ## shuffle the test pairs
            random_idx = range(len(label_verif))
            random.shuffle(random_idx)
            xx = zip(image_list_test[0::2], image_list_test[1::2], label_list_test[0::2], label_list_test[1::2],
                     label_cari_test[0::2], label_cari_test[1::2])
            yy = [xx[i] for i in random_idx]
            label_verif = [label_verif[i] for i in random_idx]
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


            paths = image_list_test
            actual_issame = label_verif


            ###### Load the model #####
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet_ext.get_model_filenames(os.path.expanduser(args.model_dir))

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet_ext.load_model(args.model_dir, meta_file, ckpt_file)


            # Get input and output tensors
            #images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch:0")
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            embeddings_visual = tf.get_default_graph().get_tensor_by_name("embeddings_expression:0")
            embeddings_cari = tf.get_default_graph().get_tensor_by_name("embeddings_cari:0")
            keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name('keep_probability:0')
            #weight_decay_placeholder = tf.get_default_graph().get_tensor_by_name('weight_decay:0')
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
            phase_train_placeholder_visual = tf.get_default_graph().get_tensor_by_name('phase_train_expression:0')
            phase_train_placeholder_cari = tf.get_default_graph().get_tensor_by_name('phase_train_cari:0')

            logits_visual = tf.get_default_graph().get_tensor_by_name('logits_expr:0')
            logits_cari = tf.get_default_graph().get_tensor_by_name('logits_cari:0')
            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
            embeddings_visual_size = embeddings_visual.get_shape()[1]
            embeddings_cari_size = embeddings_cari.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on input images')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            emb_visual_array = np.zeros((nrof_images, embeddings_visual_size))
            emb_cari_array = np.zeros((nrof_images, embeddings_cari_size))
            #emb_array = np.zeros((2*batch_size, embedding_size))
            logits_visual_size = logits_visual.get_shape()[1]
            logits_visual_array = np.zeros((nrof_images, logits_visual_size), dtype=float)
            logits_cari_size = logits_cari.get_shape()[1]
            logits_cari_array = np.zeros((nrof_images, logits_cari_size), dtype=float)
            for i in range(nrof_batches):
                print("Test batch:%d/%d\n"%(i,nrof_batches))
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet_ext.load_data(paths_batch, False, False, image_size)
                feed_dict = {phase_train_placeholder: False, phase_train_placeholder_visual: False, phase_train_placeholder_cari: False, images_placeholder:images, keep_probability_placeholder:1.0}
                #feed_dict = {phase_train_placeholder: False, images_placeholder: images}
                emb_, emb_visual_, emb_cari_, logits_visual_, logits_cari_ = sess.run([embeddings, embeddings_visual, embeddings_cari, logits_visual, logits_cari], feed_dict=feed_dict)
                emb_array[start_index:end_index,:] = emb_
                emb_visual_array[start_index:end_index,:] = emb_visual_
                emb_cari_array[start_index:end_index,:] = emb_cari_
                logits_visual_array[start_index:end_index,:] = logits_visual_
                logits_cari_array[start_index:end_index,:] = logits_cari_

               print('Evaluate_model: %s' % args.evaluate_mode)


            #### Evaluation Face verification #############
            if args.evaluate_mode == 'Euclidian':
                tprs, fprs, accuracy, val, val_std, fp_idx, fn_idx,best_threshold = lfw_ext.evaluate(emb_array, actual_issame, nrof_folds=args.lfw_nrof_folds, far=args.far)

            nrof_test_tp_pairs = sum(actual_issame)
            nrof_test_tn_pairs = len(actual_issame) - nrof_test_tp_pairs
            nrof_test_paths = len(actual_issame)*2
            # paths_pairs shape: [2, number of pairs], each column is corresponding to a pair of images
            paths_pairs = [paths[0:nrof_test_paths:2], paths[1:nrof_test_paths:2]]
            paths_pairs_array = np.array(paths_pairs)
            fp_images_paths = paths_pairs_array[:, fp_idx];
            fn_images_paths = paths_pairs_array[:, fn_idx];
            _, nrof_fp_pairs = fp_images_paths.shape
            _, nrof_fn_pairs = fn_images_paths.shape

            ################### select the false positive/negative images ####################

            print('Accuracy: %1.3f+-%1.3f\n' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f\n' % (val, val_std, args.far))

            auc = np.zeros(tprs.shape[0])
            #eer = np.zeros(tprs.shape[0])
            for i in range(tprs.shape[0]):
                fpr = fprs[i]
                tpr = tprs[i]
                auc[i] = metrics.auc(fpr, tpr)
                #eer[i] = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)

            auc_mean = np.mean(auc)
            auc_std = np.std(auc)
            # eer_mean = np.mean(eer)
            # eer_std = np.std(eer)
            print('Area Under Curve (AUC): %1.3f+-%1.3f\n' % (auc_mean, auc_std))
            #print('Equal Error Rate (EER): %1.3f+-%1.3f\n' % (eer_mean, eer_std))


            with open(os.path.join(log_dir, 'validation_on_dataset.txt'), 'at') as f:
                print('Saving the evaluation results...\n')
                f.write('arguments: %s\n--------------------\n' % ' '.join(sys.argv))
                #f.write('Identity verification acc is : %2.3f\n' % identity_verif_acc)
                f.write('Average accuracy: %1.3f+-%1.3f\n' % (np.mean(accuracy), np.std(accuracy)))
                f.write('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f\n' % (val, val_std, args.far))
                f.write('Best threshold: %2.5f \n' % (best_threshold))
                #f.write('Validation threshold: %2.5f \n' % (val_threshold))
                f.write('Area Under Curve (AUC): %1.3f+-%1.3f\n' % (auc_mean, auc_std))
                #f.write('Equal Error Rate (EER): %1.3f+-%1.3f\n' % (eer_mean, eer_std))
                f.write('------------- \n')
                #f.write('Expression recognition acc: %2.3f\n'%test_expr_acc)
                #f.write('Expression verification acc: %2.3f\n'%expression_verif_acc)
                #f.write('------------- \n')
                #f.write('Global Verification-Expression liveness acc is : %2.3f\n'%global_acc)
                #print('Saving the False positive pairs images ...\n ')
                f.write('False positive pairs: %d / %d -----------------------------------------\n' % ( \
                nrof_fp_pairs, nrof_test_tp_pairs))
                for i in range(nrof_fp_pairs):
                    f.write('%d %s\n' % (i, fp_images_paths[:, i]))
                print('Saving the False negative pairs images ...\n ')
                f.write('False negative pairs: %d / %d ---------------------------------------\n' % ( \
                nrof_fn_pairs, nrof_test_tn_pairs))
                for i in range(nrof_fn_pairs):
                    f.write('%d %s\n' % (i, fn_images_paths[:, i]))
            ################### edit by mzh 12012017: write the false positive/negative images to the file  ####################

            false_images_list = os.path.join(log_dir, 'validation_on_dataset.txt')
            save_dir = log_dir
            save_false_images.save_false_images(false_images_list, save_dir)

            with open(os.path.join(log_dir, 'validation_on_dataset.txt'), 'at') as f:
                print('Saving the tpr, fpr of ROC ...\n ')
                f.write('ROC: tpr, fpr ---------------------------------------------\n')
                for tp,fp in zip(tpr, fpr):
                    f.write('tpr/fpr: %f/%f\n'%(tp,fp))

            fig = plt.figure()
            plt.plot(fpr, tpr, label='ROC')
            plt.title('Receiver Operating Characteristics')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.plot([0, 1], [0, 1], 'g--')
            plt.grid(True)
            #plt.show()
            fig.savefig(os.path.join(log_dir, 'roc.png'))

def plot_roc(fpr, tpr, label):
    figure = plt.figure()
    plt.plot(fpr, tpr, label=label)
    plt.title('Receiver Operating Characteristics')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.plot([0, 1], [0, 1], 'g--')
    plt.grid(True)
    plt.show()

    return figure

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--model_dir', type=str,
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--test_pairs', type=str,
        help='The file containing the pairs to use for validation.')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--far', type=float,
                        help='FAR/ false acception rate (false positive rate) for evaluating the validation / recall rate', default=0.01)
    parser.add_argument('--evaluate_mode', type=str,
                        help='The evaluation mode: Euclidian distance or similarity by cosine distance.', default='Euclidian')
    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='/data/zming/logs/')
    parser.add_argument('--features_dir', type=str,
                        help='Directory where to write event logs.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
