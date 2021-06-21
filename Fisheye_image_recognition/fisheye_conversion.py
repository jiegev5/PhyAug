from wand.image import Image
from glob import glob
import os
import sys
import cv2
import numpy as np

def make_dir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)

def fisheye_conversion(source,destination,parameters):
    with Image(filename=source) as img:
        print(img.size)
        img.virtual_pixel = 'gray'
        '''
        '''
        img.distort('barrel',parameters)
        print("saving ",destination)
        img.save(filename=destination)
        return img

def cv2_img_show(fname):
    img = cv2.imread(fname)
    cv2.imshow('picture',img)
    cv2.waitKey(0) # the window will show forever
    cv2.destroyAllWindows()

def gen_random_parameters():
    param = np.random.random(size=4)
    param = param/np.sum(param) # make sure the sum is 1
    # return tuple(param)
    # return (0.15,0.15,0.2)
    # return (0.2,0.2,0.01)
    # undistort
    return (0.28441,-0.51535,-0.41022)

def gen_20_parameters():
    param_list = []
    for i in range(5):
        param = np.random.random(size=4)
        param = param/np.sum(param) # make sure the sum is 1
        param_list.append(tuple(param))
    return param_list


original_root = 'data/caltech101'
dest_root = 'data/caltech101_fisheye_dataset'
category = ['train','valid','test']

make_dir(dest_root)

train_dir = os.path.join(original_root,category[0])
train_list = os.listdir(train_dir)
print(train_list)

for cat in category:
    original_cat = os.path.join(original_root,cat) # data/caltech101_test/train
    dest_cat = os.path.join(dest_root,cat) # data/caltech101_fisheye/train
    make_dir(dest_cat)

    classes = os.listdir(original_cat)
    for c in classes:
        original_folder = os.path.join(original_cat,c) # data/caltech101_test/train/ketch
        dest_folder = os.path.join(dest_cat,c) # data/caltech101_fisheye/train/ketch
        make_dir(dest_folder)
        file_list = os.listdir(original_folder)
        key = 0
        para = gen_random_parameters()
        print("parameter to use: ",para)
        for file in file_list:
            dest_fname = os.path.join(dest_folder,file) # data/caltech101_fisheye/train/ketch/xxx.jpg
            original_fname = os.path.join(original_folder,file) # data/caltech101_test/train/ketch/xxx.jpg
            fisheye_conversion(original_fname,dest_fname,para)
            
            key = key + 1
            if key == 100: # use a new parameter for every 100 images
                key = 0
                para = gen_random_parameters()
                print("parameter to use: ",para)

            # cv2_img_show(dest_fname)
            # sys.exit()