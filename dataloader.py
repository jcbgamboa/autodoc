__author__ = 'Ayushman Dash'

import os, traceback
import numpy as np
import random
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

class Dataset :
    def __init__(self, dataset_name):
        '''
        This initializes the dataset object and creates the following variables
        1.  parent_dir: The parent directory of the dataset
        2.  target_file_path: The file that has a list of labels. It is
                    supposed to be "labels.txt" in the dataset root directory
        3.  data_root: The root directory of the data in the dataset
                    directory. This will be appended to the relative path
                    present in the train.txt, test.txt and validate.txt
        4.  train_file_path: Path of the file that contains the training
                    data. It is supposed to be "train.txt" and should be
                    placed in the root directory of the dataset. It should
                    consist of lines like
                    "<relative path to the datafile> <class label id>"
                    Make sure that the relative path and the class label are
                    separated by a "blank space" and the <class label> is an
                    index of the class label in "labels.txt".
        5.  test_file_path: Path of the file that contains the testing
                    data. It is supposed to be "test.txt" and should be
                    placed in the root directory of the dataset. It should
                    consist of lines like
                    "<relative path to the datafile> <class label id>"
                    Make sure that the relative path and the class label are
                    separated by a "blank space" and the <class label> is an
                    index of the class label in "labels.txt".
        6.  validate_file_path: Path of the file that contains the validation
                    data. It is supposed to be "validation.txt" and should be
                    placed in the root directory of the dataset. It should
                    consist of lines like
                    "<relative path to the datafile> <class label id>"
                    Make sure that the relative path and the class label are
                    separated by a "blank space" and the <class label> is an
                    index of the class label in "labels.txt".
        6.  has_train: A flag that tells if train.txt exists or not. Some
                    datasets might just have just one file which lists all the
                    data files.
        7.  has_test: A flag that tells if test.txt exists or not. Some
                    datasets might just have just one file which lists all the
                    data files
        8.  has_validate: A flag that tells if tvalidation.txt exists or not.
                    Some datasets might just have just one file which lists
                    all the data files
        9.  target: A list of all the unique labels in the dataset
        10. n_target: number of unique labels in the dataset
        11. one_hot_targets: A codebook with one-hot encodings of the
                targets. It is in the format [n_target X n_target] with the row index
                corresponding to the index of target

        :param dataset_name: name of the dataset to load.

        A folder named "dataset_name will be searched in "data/datasets" and
        the above mentioned data will be loaded

        For Example, put the dataset with the following structure in
        data/dataset/<name_of_dataset>/
                                |__train.txt
                                |__test.txt
                                |__validate.txt
                                |__labels.txt
                                |__data/
                                    |__<any_random_optional_folder>/<data_file>

        Remember that train.txt, test.txt are validate.txt are not required
        always. any one would work. Make sure you check the respective flags
        before use.
        '''
        self.gen_counter = 0
        self.data_list = None

        self.parent_dir = os.path.join("data/datasets/", dataset_name)
        self.target_file_path = os.path.join(self.parent_dir, 'labels.txt')
        self.data_root = os.path.join(self.parent_dir, 'data')
        self.train_file_path = os.path.join(self.parent_dir, 'train.txt')
        self.test_file_path = os.path.join(self.parent_dir, 'test.txt')
        self.validate_file_path = os.path.join(self.parent_dir, 'validate.txt')

        self.has_train = os.path.isfile(self.train_file_path)
        self.has_test = os.path.isfile(self.test_file_path)
        self.has_validate = os.path.isfile(self.validate_file_path)

        try:
            with open(self.target_file_path) as f :
                self.target = f.readlines()
                self.n_target = len(self.target)
        except IOError:
            print('Could not load the labels.txt file in the dataset. A '
                  'dataset folder is expected in the "data/datasets" '
                  'directory with the name that has been passed as an '
                  'argument to this method. This directory should contain a '
                  'file called labels.txt which contains a list of labels and '
                  'corresponding folders for the labels with the same name as '
                  'the labels.')
            traceback.print_stack()

        lbl_idxs = np.arange(len(self.target))
        self.one_hot_targets = np.zeros((self.n_target, self.n_target))
        self.one_hot_targets[np.arange(self.n_target), lbl_idxs] = 1

        self.batch_size = 64
        self.resize = [1000, 750]
        self.mode = 'train'
        self.model = 'caes'
        self.absolute_count = 0

    def one_hot_encode_str_lbl(self, lbl):
        '''
        Encodes a string label into one-hot encoding

        Example:
            input: "window"
            output: [0 0 0 0 0 0 1 0 0 0 0 0]
        the length would depend on the number of classes in the dataset. The
        above is just a random example.

        :param lbl: The string label
        :return: one-hot encoding
        '''
        idx = self.target.index(lbl)
        return self.one_hot_targets[idx]

    def one_hot_encode_id_lbl(self, lbl):
        '''
        Encodes a string label (id) into one-hot encoding

        Example:
            input: "13"
            output: [0 0 0 0 0 0 1 0 0 0 0 0]
        the length would depend on the number of classes in the dataset. The
        above is just a random example.

        :param lbl: The string label id
        :return: one-hot encoding
        '''
        idx = int(lbl)
        return self.one_hot_targets[idx]

    def load_data(self, batch_size=64, resize=[1000, 750], mode='train'):
        '''
        This is a generator that yields a minibatch of images in the
        following shape
        [BATCH_SIZE X NUM_CHANNELS X WIDTH X HEIGHT]

        :param batch_size: The desired batch size
        :param resize: [width X height] this is for resizing the images to a
                        fixed size.
        :param mode: To load train, test or validate data
        :return: [batch_of_images, batch_of_labels]
        '''

        img_batch = []
        lbl_batch = []
        data_list = []
        if mode == 'train':
            with open(self.train_file_path, 'rb') as data_file :
                data_list = data_file.readlines()
        elif mode == 'test' and self.has_test:
            with open(self.test_file_path, 'rb') as data_file :
                data_list = data_file.readlines()
        elif mode == 'validate' and self.has_validate:
            with open(self.validate_file_path, 'rb') as data_file :
                data_list = data_file.readlines()

        random.shuffle(data_list)

        for count, l in enumerate(data_list):
        #for count, l in list(enumerate(data_list))[0:128] :
            img_path, lbl = l.strip().split()
            try :
                if (count % (batch_size)) == 0 and count != 0 :
                   yield (np.array(img_batch), np.array(lbl_batch))
                   img_batch = []
                   lbl_batch = []
                img, one_hot_lbl = self.read_data(img_path, lbl, resize)
                img_batch.append(img)
                lbl_batch.append(one_hot_lbl)
            except IOError:
                print('There was an error while yielding a minibatch from the '
                      'dataset. Please make sure that you have followed the'
                      ' instructions properly and try again.')
                traceback.print_stack()
                pass

    def __iter__(self):
        return self

    def next(self):
        '''

        :return:
        '''

        img_batch = []
        lbl_batch = []
        if self.data_list is None:
            if self.mode == 'train':
                with open(self.train_file_path, 'rb') as data_file :
                    self.data_list = data_file.readlines()
            elif self.mode == 'test' and self.has_test:
                with open(self.test_file_path, 'rb') as data_file :
                    self.data_list = data_file.readlines()
            elif self.mode == 'validate' and self.has_validate:
                with open(self.validate_file_path, 'rb') as data_file :
                    self.data_list = data_file.readlines()
            random.shuffle(self.data_list)

        batch_file_data = self.data_list[self.gen_counter:
					(self.gen_counter + self.batch_size)]

        if self.gen_counter + self.batch_size > len(self.data_list):
            self.gen_counter = 0

        for count, l in enumerate(batch_file_data):
        #for count, l in list(enumerate(data_list))[0:128] :
            img_path, lbl = l.strip().split()
            try :
                self.absolute_count += 1
                img, one_hot_lbl = self.read_data(img_path, lbl, self.resize)
                img_batch.append(img)
                lbl_batch.append(one_hot_lbl)
            except IOError:
                print('There was an error while yielding a minibatch from the '
                      'dataset. Please make sure that you have followed the'
                      ' instructions properly and try again.')
                traceback.print_stack()

        self.gen_counter += self.batch_size
        return (np.array(img_batch), np.array(lbl_batch))

    def __next__(self):
        return self.load_data()

    def is_grey_scale(img_path) :
        '''
        A utility method to be used later to check whether an image is
        grayscale of color
        :return: True if gray scale
        '''
        im = Image.open(img_path).convert('RGB')
        w, h = im.size
        for i in range(w) :
            for j in range(h) :
                r, g, b = im.getpixel((i, j))
                if r != g != b :
                    return False
        return True

    def read_data(self, img_path, lbl, resize):
        '''
        This method reads an image and then preprocesses it.
        :param img_path: relative path to the image
        :param lbl: the class label id
        :param resize: [width, height] to resize the image
        :return: image, one_hot_label
        '''

        with open(os.path.join(self.data_root, img_path), 'rb') as img_file :
            img = Image.open(img_file)
            w, h = img.size

            img = img.resize((resize[1], resize[0]), Image.BICUBIC)

            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype = np.uint8)
            #plt.imshow(img_bw, cmap='gray')
            #plt.show()
            if self.model is 'caes':
                img_bw = img_bw[np.newaxis, :]
            else:
                img_bw = img_bw[:,:, np.newaxis]

        one_hot_lbl = self.one_hot_encode_id_lbl(lbl)
        return img_bw, one_hot_lbl


