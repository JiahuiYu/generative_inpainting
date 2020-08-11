# 将原数据集分为training ,validation  by gavin
import os
import random

import argparse

# 划分验证集训练集
_NUM_TEST = 2800

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/work/r08922a13/Waveform-auto-encoder/datasets/ESC-50-master/spectrogram_cut256_aug', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./data/esc50/cut256_aug/train_shuffled.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data/esc50/cut256_aug/validation_static_view.flist', type=str,
                    help='The validation filename.')


def _get_filenames(dataset_dir):
    photo_filenames = []
    image_list = os.listdir(dataset_dir)
    photo_filenames = [os.path.join(dataset_dir, _) for _ in image_list]
    return photo_filenames


if __name__ == "__main__":

    args = parser.parse_args()
    data_dir = args.folder_path

    # get all file names
    photo_filenames = _get_filenames(data_dir)
    print("size of esc50 is %d" % (len(photo_filenames)))

    # 切分数据为测试训练集
    random.seed(0)
    random.shuffle(photo_filenames)
    training_file_names = photo_filenames[_NUM_TEST:]
    validation_file_names = photo_filenames[:_NUM_TEST]

    print("training file size:",len(training_file_names))
    print("validation file size:", len(validation_file_names))

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    if not os.path.exists(args.validation_filename):
        os.mknod(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename)
