import os
import shutil
import numpy as np

import errno
import os
import cv2
from PIL import Image

def resize_in_path(width=64, height=64, path="training_data/validation/textures_small/"):
    dirs = os.listdir(path)
    for item in dirs:
        im = Image.open(path + item)
        f, e = os.path.splitext(path + item)
        imResize = im.resize((width, height), Image.ANTIALIAS)
        imResize.save(f + '_.png', 'png', quality=100)

def create_if_does_not_exist_2(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

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


#convert_formats("data_sets_our/textures/train")
#convert_formats("data_sets_our/textures/test")
#convert_formats("data_sets_our/textures/validation")


def split_to_training_and_test_sets_given_path(src_path, test_path, training_path, test_percent = 0.2):
    files = os.listdir(src_path)
    print(len(files))
    create_if_does_not_exist_2(test_path)
    create_if_does_not_exist_2(training_path)
    
    
    for f in files:
        if np.random.rand(1) < test_percent:
            shutil.move(os.path.join(src_path,f), os.path.join(test_path, f))
        else:
            shutil.move(os.path.join(src_path,f), os.path.join(training_path, f))

a = os.getcwd()
print(a)
src_path = os.path.join(a,'training_data')
test_path = os.path.join(src_path,'validation')
training_path = os.path.join(src_path,'train')
split_to_training_and_test_sets_given_path(src_path, test_path, training_path)

def open_path_or_create(path_dir):
    if not os.path.exists(os.path.dirname(path_dir)):
        try:
            os.makedirs(os.path.dirname(path_dir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

# def get_dims(src):
#     a = cv2.imread(src, cv2.IMREAD_ANYCOLOR)
#     print(a)
#
# get_dims("training_data/training/textures/1.png")
