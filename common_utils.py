import os
import shutil
import numpy as np

import errno
import os
import cv2
from PIL import Image


def resize_in_path(width=64, height=64, path="./"):
    dirs = os.listdir(path)
    for item in dirs:
        im = Image.open(path + item)
        f, e = os.path.splitext(path + item)
        im_resize = im.resize((width, height), Image.ANTIALIAS)
        im_resize.save(f + e, quality=100)


def create_path_if_not_exists(path_dir):
    if not os.path.exists(path_dir):
        try:
            os.makedirs(path_dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def convert_formats(mypath, prefix=""):
    all_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        all_filenames.extend(filenames)
        break
    print("txt_name_list", all_filenames)

    """ Process """
    i = 1
    for filename in all_filenames:
        full_path = os.path.join(mypath, filename)
        new_full_path = os.path.join(mypath, prefix + str(i) + '.png')
        os.rename(full_path, new_full_path)
        i += 1


def split_to_training_and_validation(src_path, validation_path=None, training_path=None, test_percent=0.2):
    if (validation_path is None):
        validation_path = os.path.join(src_path, "validation")
    if (training_path is None):
        training_path = os.path.join(src_path, "training")
    files = os.listdir(src_path)
    print("number of files: " + str(len(files)))
    create_path_if_not_exists(validation_path)
    create_path_if_not_exists(training_path)

    for f in files:
        if np.random.rand(1) < test_percent:
            shutil.move(os.path.join(src_path, f), os.path.join(validation_path, f))
        else:
            shutil.move(os.path.join(src_path, f), os.path.join(training_path, f))


def doo():
    # here we use the util functions when needed
    resize_in_path(64, 64, "./test_data/")
    return


doo()

####### comments #######

# convert_formats("data_sets_our/textures/train")
# convert_formats("data_sets_our/textures/test")
# convert_formats("data_sets_our/textures/validation")

# a = os.getcwd()
# print(a)
# src_path = os.path.join(a,'training_data')
# test_path = os.path.join(src_path,'validation')
# training_path = os.path.join(src_path,'training')
# split_to_training_and_test_sets_given_path(src_path, test_path, training_path)

# def get_dims(src):
#     a = cv2.imread(src, cv2.IMREAD_ANYCOLOR)
#     print(a)
#
# get_dims("training_data/training/textures/1.png")
