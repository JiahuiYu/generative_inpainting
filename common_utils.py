import math
import shutil
import numpy as np
import errno
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random


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


def generate_mask_img(dest_dir, mask_w, mask_h, img_w, img_h, i, j, name="mask.png"):
    mask_img = np.zeros((img_h, img_w), dtype=bool)
    mask_img[i:i + mask_h, j:j + mask_w] = np.ones((mask_h, mask_w), dtype=bool)
    # save mask
    mask_path = os.path.join(dest_dir, name)
    plt.imsave(mask_path, mask_img, cmap=cm.gray)
    return mask_img


def get_default_mask_top_left_corner(mask_w, mask_h, img_w, img_h):
    i = math.floor(img_h / 2) - math.floor(mask_h / 2)
    j = math.floor(img_w / 2) - math.floor(mask_w / 2)
    return (i, j)


class Mask:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def create_test_cases(base_dir, h_limit, w_limit, test_data_dir, num_img_test, num_masks):
    imgs = os.listdir(test_data_dir)
    create_path_if_not_exists(base_dir)
    masks = []
    for i in range(num_masks):
        satisfied = False
        curr = None
        while (not satisfied):
            curr = create_random_mask(h_limit, w_limit)
            # prevent very small masks
            satisfied = curr.width >= 10 and curr.height >= 10
        masks.append(curr)
    # create mask directories
    for mask in masks:
        mask_id = "m[{}_{}][{}_{}]".format(mask.top, mask.left, mask.height, mask.width)
        mask_dir = os.path.join(base_dir, mask_id)
        create_path_if_not_exists(mask_dir)
        plt.imsave(mask_dir, mask.mask_img, cmap=cm.gray)
        for i in range(num_img_test):
            img_name = imgs[i]
            img = plt.imread(os.path.join(test_data_dir, img_name))
            img[mask.mask_img] = 1
            # save incomplete image
            plt.imsave(os.path.join(mask_dir, img_name), img)


def create_random_mask(h_limit, w_limit, margin=5, is_square=False, is_show=False):
    top = random.randint(margin, w_limit - margin)
    left = random.randint(margin, h_limit - margin)
    mask_w = random.randint(0, w_limit - left - margin)
    if is_square:
        mask_h = mask_w
    else:
        mask_h = random.randint(0, h_limit - top - margin)
    return create_mask(top, left, mask_h, mask_w, h_limit, w_limit, is_show)


def create_mask(top, left, mask_h, mask_w, h_limit, w_limit, is_show=False):
    mask_img = np.zeros((h_limit, w_limit), dtype=bool)
    mask_img[top:top + mask_h, left:left + mask_w] = np.ones((mask_h, mask_w), dtype=bool)
    if (is_show):
        plt.imshow(mask_img, cmap=plt.cm.gray)  # use appropriate colormap here
        plt.show()
    mask = Mask(mask_img=mask_img, top=top, left=left, width=mask_w, height=mask_h)
    return mask


def create_test_imgs():
    create_test_cases(
        test_data_dir="test_data",
        h_limit=64,
        w_limit=64,
        num_img_test=10,
        base_dir="test_cases",
        num_masks=10
    )
    return


def get_files_in_dir(d, full_path=False):
    if (full_path):
        # gets full relative path of all files in directory
        return [os.path.join(d, o) for o in os.listdir(d) if os.path.isfile(os.path.join(d, o))]
    return [o for o in os.listdir(d) if os.path.isfile(os.path.join(d, o))]


ORIGINAL_WEIGHTS_64_64_100000_ITERS = "model_logs/20180805072004313626_arik-olsh-gpu_celeba_NORMAL_wgan_gp_placesa_small_log_dir"  # /snap-80000"


def run_test():
    test_dir = "test_cases"
    test_name = "run1"
    checkpoint_dir = ORIGINAL_WEIGHTS_64_64_100000_ITERS
    mask_files = get_files_in_dir(test_dir, full_path=True)
    for mask in mask_files:
        test_imgs_dir, _ = os.path.splitext(mask)
        test_results_dir = os.path.join(test_imgs_dir, test_name)
        create_path_if_not_exists(test_results_dir)
        test_imgs = get_files_in_dir(test_imgs_dir)
        for img in test_imgs:
            full_img_path = os.path.join(test_imgs_dir, img)
            output = os.path.join(test_results_dir, img)
            cmd_pattern = "python test.py --image {} --mask {} --output {} --checkpoint {}"
            cmd = cmd_pattern.format(full_img_path, mask, output, checkpoint_dir)
            print("running %s" % cmd)
            os.system(cmd)
    return


# create_test_imgs()
run_test()

####### comments #######

# to test model:
# 1. run create_test_imgs() to create folders with different masks and incomplete images
# 2. run run_test() to run the model on these folders with the right parameters.
