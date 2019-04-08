

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
#import facenet
#import lfw
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
import  save_false_images_webcavi
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
            ###### Load the model #####
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet_ext.get_model_filenames(os.path.expanduser(args.model_dir))

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet_ext.load_model(args.model_dir, meta_file, ckpt_file)
            Rank_1_acc = []
            Rank_10_acc = []

            K = args.lfw_nrof_folds
            for k in range(K):
                probe = os.path.join(args.probe, 'FR_Probe_C2P%d'%(k+1)+'.txt')
                gallery = os.path.join(args.gallery, 'FR_Gallery_C2P%d'%(k+1)+'.txt')
                # probe = os.path.join(args.probe, 'FR_Probe_P2C%d'%(k+1)+'.txt')
                # gallery = os.path.join(args.gallery, 'FR_Gallery_P2C%d'%(k+1)+'.txt')
                rank_1_acc, rank_10_acc = rank_n(probe, gallery, args.data_dir, args.lfw_batch_size, sess)

                Rank_1_acc.append(rank_1_acc)
                Rank_10_acc.append(rank_10_acc)

            Rank1_mean = np.mean(np.array(Rank_1_acc))
            Rank1_std = np.std(np.array(Rank_1_acc))
            Rank10_mean = np.mean(np.array(Rank_10_acc))
            Rank10_std = np.std(np.array(Rank_10_acc))
            print('Rank1 accuracy: %1.3f+-%1.3f\n' % (Rank1_mean, Rank1_std))
            print('Rank2 accuracy: %1.3f+-%1.3f\n' % (Rank10_mean, Rank10_std))

            with open(os.path.join(log_dir, 'Rank_on_dataset.txt'), 'at') as f:
                print('Saving the evaluation results...\n')
                f.write('arguments: %s\n--------------------\n' % ' '.join(sys.argv))
                #f.write('Identity verification acc is : %2.3f\n' % identity_verif_acc)
                f.write('Rank1 accuracy: %1.3f+-%1.3f\n' %(Rank1_mean, Rank1_std))
                f.write('Rank2 accuracy: %1.3f+-%1.3f\n' %(Rank10_mean, Rank10_std))

def rank_n(probe, gallery, data_dir, lfw_batch_size, sess):
    #########################   webcari    ##########################
    image_list_probe, label_list_id_probe \
        = facenet_ext.get_image_paths_and_labels_webcari_rank(probe, data_dir)

    image_list_gallery, label_list_id_gallery \
        = facenet_ext.get_image_paths_and_labels_webcari_rank(gallery, data_dir)

    ## mapping the string id label to the number id label
    see_id = list(set(label_list_id_probe + label_list_id_gallery))
    see_id.sort()
    See_id = []
    for id in see_id:
        See_id.append(id.upper())

    label_tmp = []
    for label in label_list_id_probe:
        label_tmp.append(See_id.index(label.upper()))
    label_list_id_probe = label_tmp

    label_tmp = []
    for label in label_list_id_gallery:
        label_tmp.append(See_id.index(label.upper()))
    label_list_id_gallery = label_tmp

    paths = image_list_probe

    # Get input and output tensors
    # images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch:0")
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    embeddings_visual = tf.get_default_graph().get_tensor_by_name("embeddings_expression:0")
    embeddings_cari = tf.get_default_graph().get_tensor_by_name("embeddings_cari:0")
    keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name('keep_probability:0')
    # weight_decay_placeholder = tf.get_default_graph().get_tensor_by_name('weight_decay:0')
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
    print('Runnning forward pass on probe images')
    batch_size = lfw_batch_size
    nrof_images = len(paths)
    nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))
    emb_visual_array = np.zeros((nrof_images, embeddings_visual_size))
    emb_cari_array = np.zeros((nrof_images, embeddings_cari_size))
    # emb_array = np.zeros((2*batch_size, embedding_size))
    logits_visual_size = logits_visual.get_shape()[1]
    logits_visual_array = np.zeros((nrof_images, logits_visual_size), dtype=float)
    logits_cari_size = logits_cari.get_shape()[1]
    logits_cari_array = np.zeros((nrof_images, logits_cari_size), dtype=float)
    for i in range(nrof_batches):
        print("Test batch:%d/%d\n" % (i, nrof_batches))
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet_ext.load_data(paths_batch, False, False, image_size)
        feed_dict = {phase_train_placeholder: False, phase_train_placeholder_visual: False,
                     phase_train_placeholder_cari: False, images_placeholder: images, keep_probability_placeholder: 1.0}
        # feed_dict = {phase_train_placeholder: False, images_placeholder: images}
        emb_, emb_visual_, emb_cari_, logits_visual_, logits_cari_ = sess.run(
            [embeddings, embeddings_visual, embeddings_cari, logits_visual, logits_cari], feed_dict=feed_dict)
        emb_array[start_index:end_index, :] = emb_
        emb_visual_array[start_index:end_index, :] = emb_visual_
        emb_cari_array[start_index:end_index, :] = emb_cari_
        logits_visual_array[start_index:end_index, :] = logits_visual_
        logits_cari_array[start_index:end_index, :] = logits_cari_

    emb_array_probe = emb_array

    paths = image_list_gallery
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on gallery images')
    batch_size = lfw_batch_size
    nrof_images = len(paths)
    nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))
    emb_visual_array = np.zeros((nrof_images, embeddings_visual_size))
    emb_cari_array = np.zeros((nrof_images, embeddings_cari_size))
    # emb_array = np.zeros((2*batch_size, embedding_size))
    logits_visual_size = logits_visual.get_shape()[1]
    logits_visual_array = np.zeros((nrof_images, logits_visual_size), dtype=float)
    logits_cari_size = logits_cari.get_shape()[1]
    logits_cari_array = np.zeros((nrof_images, logits_cari_size), dtype=float)
    for i in range(nrof_batches):
        print("Test batch:%d/%d\n" % (i, nrof_batches))
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet_ext.load_data(paths_batch, False, False, image_size)
        feed_dict = {phase_train_placeholder: False, phase_train_placeholder_visual: False,
                     phase_train_placeholder_cari: False, images_placeholder: images, keep_probability_placeholder: 1.0}
        # feed_dict = {phase_train_placeholder: False, images_placeholder: images}
        emb_, emb_visual_, emb_cari_, logits_visual_, logits_cari_ = sess.run(
            [embeddings, embeddings_visual, embeddings_cari, logits_visual, logits_cari], feed_dict=feed_dict)
        emb_array[start_index:end_index, :] = emb_
        emb_visual_array[start_index:end_index, :] = emb_visual_
        emb_cari_array[start_index:end_index, :] = emb_cari_
        logits_visual_array[start_index:end_index, :] = logits_visual_
        logits_cari_array[start_index:end_index, :] = logits_cari_

    emb_array_gallery = emb_array

    dist_emb_prob_array = np.zeros((emb_array_probe.shape[0], emb_array_gallery.shape[0]))
    for i in range(emb_array_probe.shape[0]):
        emb_prob = emb_array_probe[i]
        for j in range(emb_array_gallery.shape[0]):
            emb_gallery = emb_array_gallery[j]
            diff = np.subtract(emb_prob, emb_gallery)
            dist = np.sum(np.square(diff))
            dist_emb_prob_array[i][j] = dist

    rank1_idx = np.argmin(dist_emb_prob_array, 1)
    label_list_id_probe_predict = [label_list_id_gallery[i] for i in rank1_idx]
    correct_prediction = np.equal(label_list_id_probe_predict, label_list_id_probe)
    rank1_acc = np.mean(correct_prediction)
    print('Rank 1 acc: %2.3f' % rank1_acc)

    n = 10
    label_list_id_probe_predict = []
    rank_n_idx = np.argpartition(dist_emb_prob_array, n, 1)
    for i in range(rank_n_idx.shape[0]):
        label_list_id_probe_predict += [label_list_id_gallery[i] for i in rank_n_idx[i][:n]]
    correct_prediction = np.equal(label_list_id_probe_predict, list(np.array(label_list_id_probe).repeat(n)))
    corr_n = 0
    for i in range(len(label_list_id_probe)):
        corr_ = correct_prediction[i::n]
        if sum(corr_) > 0:
            corr_n += 1
    rank2_acc = corr_n / len(label_list_id_probe)
    print('Rank 2 acc: %2.3f' % rank2_acc)

    return rank1_acc, rank2_acc

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
    parser.add_argument('--probe', type=str,
                        help='The file containing the pairs to use for validation.')
    parser.add_argument('--gallery', type=str,
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
                        help='Directory where to write event logs.', default='/data/zming/logs/facenet')
    parser.add_argument('--features_dir', type=str,
                        help='Directory where to write event logs.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
