import argparse
import os
import common_utils
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./data_flist/train_shuffled.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data_flist/validation_shuffled.flist', type=str,
                    help='The validation filename.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to be shuffled')

if __name__ == "__main__":

    args = parser.parse_args()

    # get the list of directories and separate them into 2 types: training and validation
    training_dirs = os.listdir(args.folder_path + "/training")
    validation_dirs = os.listdir(args.folder_path + "/validation")

    # make 2 lists to save file paths
    training_file_names = []
    validation_file_names = []

    # append all files into 2 lists
    for training_dir in training_dirs:
        # append each file into the list file names
        training_folder = os.listdir(args.folder_path + "/training" + "/" + training_dir)
        for training_item in training_folder:
            # modify to full path -> directory
            training_item = args.folder_path + "/training" + "/" + training_dir + "/" + training_item
            training_file_names.append(training_item)

    # append all files into 2 lists
    for validation_dir in validation_dirs:
        # append each file into the list file names
        validation_folder = os.listdir(args.folder_path + "/validation" + "/" + validation_dir)
        for validation_item in validation_folder:
            # modify to full path -> directory
            validation_item = args.folder_path + "/validation" + "/" + validation_dir + "/" + validation_item
            validation_file_names.append(validation_item)

    # print all file paths
    for i in training_file_names:
        print(i)
    for i in validation_file_names:
        print(i)

    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(training_file_names)
        shuffle(validation_file_names)

    # make output file if not existed
    common_utils.create_path_if_not_exists(args.train_filename)
    common_utils.create_path_if_not_exists(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)
