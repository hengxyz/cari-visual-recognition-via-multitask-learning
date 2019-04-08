"""Generating the test paris for the face verification of the youtubeface dataset.
"""
# La rochelle universite
# Copyright (c) 22/12/2016 Zuheng Ming
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import argparse

import os
import sys
import math
import itertools
import random
import shutil



def save_false_images(false_images_list, save_dir):

    # false_img_list = os.path.expanduser(args.false_images_list);
    # save_img_dir = os.path.expanduser(args.save_dir);
    false_img_list = os.path.expanduser(false_images_list);
    save_img_dir = os.path.expanduser(save_dir);
    print('False images list is : %s\n'%false_img_list)
    print('Save the false images to : %s\n' %save_img_dir)

    false_images_dir = os.path.join(save_img_dir,'false_images')
    fp_images_dir = os.path.join(false_images_dir, 'false_positive')
    fn_images_dir = os.path.join(false_images_dir, 'false_negative')
    if os.path.exists(false_images_dir):
        shutil.rmtree(false_images_dir)
        os.mkdir(false_images_dir)
        os.mkdir(fp_images_dir)
        os.mkdir(fn_images_dir)
    else:
        os.mkdir(false_images_dir)
        os.mkdir(fp_images_dir)
        os.mkdir(fn_images_dir)

    image_paris = []
    with open(false_img_list, 'r') as f:
        for line in f.readlines()[4:]:
 #            pair = line.strip(str(i/2))
 #            pair = pair.strip()
 #            pair = pair.strip('[')
 # #           pair = pair.strip(']')
 #            pair = pair.strip()
 #            pair = pair.split()
            image_paris.append(line)

    for i in range(len(image_paris)):
        strtmp = image_paris[i]
        if strtmp[0]=='F':
            if strtmp[6] == 'p':
                fp_start = i+1
            if strtmp[6] == 'n':
                fp_end = i-1
                fn_start = i+1
                fn_end = len(image_paris)-1
                break

    fp_image_paris = image_paris[fp_start:fp_end+1]
    fn_image_paris = image_paris[fn_start:fn_end+1]

    fp_imgs = []
    fn_imgs = []

    for i in range(len(fp_image_paris)):
        fp_img = fp_image_paris[i]
        fp_img = fp_img.strip(str(i/2))
        fp_img = fp_img.strip()
        fp_img = fp_img.strip('[')
        fp_img = fp_img.strip()
        fp_img = fp_img.split()
        fp_imgs.append(fp_img)

    for i in range(len(fn_image_paris)):
        fn_img = fn_image_paris[i]
        fn_img = fn_img.strip(str(i/2))
        fn_img = fn_img.strip()
        fn_img = fn_img.strip('[')
        fn_img = fn_img.strip()
        fn_img = fn_img.split()
        fn_imgs.append(fn_img)

    for i in range(0,len(fp_imgs),2):
        fp_img_pair = []
        img = fp_imgs[i][0]
        img = img[1:-1]
        #print('FP: %d  %s\n' % (i/2 + 1, img))
        fp_img_pair.append(img)
        img = fp_imgs[i+1][0]
        img = img[1:-2]
        #print('       %s\n' % (img))
        fp_img_pair.append(img)
        dstdir = os.path.join(fp_images_dir, str(int(i / 2)))
        os.mkdir(dstdir)
        for fp_img in  fp_img_pair:
            shutil.copy(fp_img, dstdir)

    for i in range(0,len(fn_imgs),2):
        fn_img_pair = []
        img = fn_imgs[i][0]
        img = img[1:-1]
        #print('FN: %d  %s\n' % (i/2 + 1, img))
        fn_img_pair.append(img)
        img = fn_imgs[i+1][0]
        img = img[1:-2]
        #print('        %s\n' % (img))
        fn_img_pair.append(img)
        dstdir = os.path.join(fn_images_dir, str(int(i / 2)))
        os.mkdir(dstdir)
        for fn_img in  fn_img_pair:
            shutil.copy(fn_img, dstdir)

    return




def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--false_images_list', type=str,
        help='Path to the false positive/negative images list directory.')
    parser.add_argument('--save_dir', type=str,
        help='Path to save the false positive/negative images directory.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    args=parse_arguments(sys.argv[1:])
    false_img_list = os.path.expanduser(args.false_images_list);
    save_img_dir = os.path.expanduser(args.save_dir);
    save_false_images(false_img_list, save_img_dir)



