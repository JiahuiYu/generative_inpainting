import math
import shutil
import numpy as np
import errno
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def generate_mask_img(dest_dir, mask_w, mask_h, img_w, img_h, i, j):
    mask_img = np.zeros((img_h, img_w), dtype=bool)
    mask_img[i:i + mask_h, j:j + mask_w] = np.ones((mask_h, mask_w), dtype=bool)
    # save mask
    mask_path = os.path.join(dest_dir, 'mask[{}_{}][{}_{}].png'.format(i, j, mask_h, mask_w))
    plt.imsave(mask_path, mask_img, cmap=cm.gray)
    return mask_img


def apply_mask_img_to_img(img, mask_img):
    img[mask_img] = 1


def apply_mask_to_img(img, mask_w, mask_h, corner_x, corner_y):
    img[corner_y:corner_y + mask_h, corner_x:corner_x + mask_w, :] = 1


def get_default_mask_top_left_corner(mask_w, mask_h, img_w, img_h):
    i = math.floor(img_h / 2) - math.floor(mask_h / 2)
    j = math.floor(img_w / 2) - math.floor(mask_w / 2)
    return (i, j)


def generate_mask_and_masked_image(img_path, mask_w, mask_h, top=None, left=None, gen_img_mask=False):
    img = plt.imread(img_path)
    [img_h, img_w, _] = img.shape

    if (top is None and left is None):
        [i, j] = get_default_mask_top_left_corner(mask_w, mask_h, img_w, img_h)
    else:
        i = top
        j = left

    if (i < 0 or i + mask_h >= img_h or j < 0 or j + mask_w >= img_w):
        raise ValueError("dimensions of mask out of img bounds")

    if (gen_img_mask):
        mask_img = generate_mask_img(os.path.dirname(img_path), mask_w, mask_h, img_w, img_h, i, j)

    apply_mask_to_img(img, mask_w, mask_h, j, i)
    # save corrupted image
    f, e = os.path.splitext(img_path)
    plt.imsave(f + '[{}_{}][{}_{}]'.format(i, j, mask_h, mask_w) + e, img)


def mask_images_demo():
    generate_mask_and_masked_image("test_data/1.png", 20, 20, gen_img_mask=True)
    generate_mask_and_masked_image("test_data/2.png", 20, 20)
    generate_mask_and_masked_image("test_data/3.png", 20, 20)
    generate_mask_and_masked_image("test_data/4.png", 20, 20)
    generate_mask_and_masked_image("test_data/5.png", 20, 20)


def doo():
    # here we use the util functions when needed
    mask_images_demo()
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
