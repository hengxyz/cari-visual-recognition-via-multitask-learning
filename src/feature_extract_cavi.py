

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
import  save_false_images
import facenet_ext
import lfw_ext
from itertools import compress

def main(args):

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    print('log_dir: %s\n' % log_dir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            

            ### all samples CAVI
            image_list_all, label_list_id_all, label_list_cari_all, nrof_classes_all = facenet_ext.get_image_paths_and_labels_cavi(
                args.data_dir)

            paths = image_list_all
            labels_actual = label_list_id_all

            # Load the model
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

            len_emb = emb_array.shape[0]
            label_list_cari_all_ = label_list_cari_all[:len_emb]
            filter_visual = [x==0 for x in label_list_cari_all_]
            filter_cari = [x==1 for x in label_list_cari_all_]

            row_visual = list(compress(range(len_emb), filter_visual))
            row_cari = list(compress(range(len_emb), filter_cari))

            emb_visual_array_real = emb_visual_array[row_visual]
            emb_c2v_array = emb_visual_array[row_cari]
            emb_cari_array_real = emb_cari_array[row_cari]
            emb_v2c_array = emb_cari_array[row_visual]

            label_id_emb_visual_real = list(np.array(label_list_id_all)[row_visual])
            label_id_emb_c2v = list(np.array(label_list_id_all)[row_cari])
            label_id_emb_cari_real = list(np.array(label_list_id_all)[row_cari])
            label_id_emb_v2c = list(np.array(label_list_id_all)[row_visual])

            np.save(os.path.join(log_dir, 'features_emb_visual_real.npy'), emb_visual_array_real)
            np.save(os.path.join(log_dir, 'features_emb_cari_real.npy'), emb_cari_array_real)
            np.save(os.path.join(log_dir, 'features_emb_c2v.npy'), emb_c2v_array)
            np.save(os.path.join(log_dir, 'features_emb_v2c.npy'), emb_v2c_array)

            np.save(os.path.join(log_dir, 'label_id_emb_visual_real.npy'), label_id_emb_visual_real)
            np.save(os.path.join(log_dir, 'label_id_emb_cari_real.npy'), label_id_emb_cari_real)
            np.save(os.path.join(log_dir, 'label_id_emb_c2v.npy'), label_id_emb_c2v)
            np.save(os.path.join(log_dir, 'label_id_emb_v2c.npy'), label_id_emb_v2c)

    return



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
    parser.add_argument('--evaluate_mode', type=str,
                        help='The evaluation mode: Euclidian distance or similarity by cosine distance.', default='Euclidian')
    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='/data/zming/logs/facenet')
    parser.add_argument('--features_dir', type=str,
                        help='Directory where to write event logs.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
