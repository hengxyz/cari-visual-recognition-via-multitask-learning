"""Functions for getting the face of the dataset webcari.
"""
# MIT License
#
# Copyright (c) 2019 Zuheng Ming
import os
import cv2
import matplotlib.pyplot as plot
import glob

original_images = '/data/zming/datasets/WebCaricature/OriginalImages'
landmarks_files = '/data/zming/datasets/WebCaricature/FacialPoints'
face_images = '/data/zming/datasets/WebCaricature/FaceDetected'
face_bb = '/data/zming/datasets/WebCaricature/FacialBB'
face_landmarks = '/data/zming/datasets/WebCaricature/Face_landmarks'
bb_scale_vertical = 1.2
bb_scale_horizontal = 1.5
def main():
    images_folders = os.listdir(original_images)
    images_folders.sort()

    for id in images_folders:
        images_id = glob.glob(os.path.join(original_images, id, '*.jpg'))
        images_id.sort()

        if not os.path.isdir(os.path.join(face_bb, id)):
            os.mkdir(os.path.join(face_bb, id))
        if not os.path.isdir(os.path.join(face_images, id)):
            os.mkdir(os.path.join(face_images, id))
        if not os.path.isdir(os.path.join(face_landmarks, id)):
            os.mkdir(os.path.join(face_landmarks, id))

        for imgfile in images_id:
            print(imgfile)
            img_name = str.split(imgfile,'/')[-1]
            img_name = str.split(img_name,'.')[0]
            img = cv2.imread(imgfile)
            ysize, xsize, _ = img.shape
            img_ld_file =  os.path.join(landmarks_files, id, img_name+'.txt')
            with open(img_ld_file, 'r') as f:
                X = []
                Y = []
                lines = f.readlines()
                for line in lines:
                    x,y=str.split(line, ' ')
                    y = y[:-2]
                    x = int(float(x)) ## float(x) is for converting the scientific notation number
                    y = int(float(y))
                    X.append(x)
                    Y.append(y)

                # # plot landmarks
                # plot.figure()
                # img1=plot.imread(imgfile)
                # plot.imshow(img1)
                # plot.plot(X,Y,'x')
                # #plot.show()
                # plot.savefig(os.path.join(face_landmarks,id,img_name+'.jpg'))
                # plot.close()

            x_min = min(X)
            x_max = max(X)
            y_min = min(Y)
            y_max = max(Y)

            x_c = int((x_min + x_max) / 2)
            y_c = int((y_min + y_max) / 2)

            x_width = int(x_max - x_min)
            y_height = int(y_max - y_min)

            x_width_scale = x_width*bb_scale_horizontal
            y_height_scale = y_height * bb_scale_vertical

            x_lf = max(0, x_c-int(x_width_scale/2))
            y_lf = max(0, y_c - int(y_height_scale / 2))
            x_rb = min(xsize, x_c + int(x_width_scale / 2))
            y_rb = min(ysize, y_c + int(y_height_scale / 2))

            with open(os.path.join(face_bb, id, img_name+'.txt'),'w') as f:
                f.write("%d %d %d %d"%(x_lf,y_lf,x_rb,y_rb))

            face = img[y_lf:y_rb,x_lf:x_rb,:]
            cv2.imwrite(os.path.join(face_images,id,img_name+'.jpg'),face)

            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                cv2.circle(img, (x,y), 3, (0,255,0), -1)
            cv2.imwrite(os.path.join(face_landmarks, id, img_name + '.jpg'), face)


    return















if __name__=='__main__':
    main();