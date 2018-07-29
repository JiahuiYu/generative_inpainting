import errno
import os
import cv2

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
