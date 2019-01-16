# -*- coding: UTF-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
import tqdm
from sklearn import model_selection


def read_image(filename, IMAGE_H, IMAGE_W, LABEL_LENGTH):
    image = cv2.imread(filename)  # 读 PNG 只会读到3个通道
    # image = mpimg.imread(filename)
    h, w = image.shape[:2]
    image = image[0:h, 12:w - 6]
    image = cv2.resize(image, (IMAGE_W * LABEL_LENGTH, IMAGE_H), cv2.INTER_LINEAR)  # 缩放大小

    # Convert from [0, 255] -> [0.0, 1.0].
    image = image.astype(np.float32)
    image = image / 255.0

    return image


def split_image(image, IMAGE_H, IMAGE_W, LABEL_LENGTH):
    images = []
    h = image.shape[0]
    sw = IMAGE_W
    for i in range(LABEL_LENGTH):
        x = sw * i
        images.append(image[0:h, x:x + sw])

    return images


# 验证码去燥
def remove_noise(image):
    return image


def load_data(path, IMAGE_H, IMAGE_W, LABEL_LENGTH, LABELS):
    # OneHot
    def char_to_vec(c):
        y = np.zeros((len(LABELS),))
        y[LABELS.index(c)] = 1.0
        return y

    labels = []
    images = []
    sfiles = []
    fnames = os.listdir(path)
    with tqdm.tqdm(total=len(fnames)) as pbar:
        for i, name in enumerate(fnames):
            if name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png"):
                image = read_image(os.path.join(path, name), IMAGE_H, IMAGE_W, LABEL_LENGTH)

                simgs = split_image(image, IMAGE_H, IMAGE_W, LABEL_LENGTH)
                label = name[:LABEL_LENGTH].upper()

                for k in range(LABEL_LENGTH):
                    labels.append(char_to_vec(label[k]))
                    images.append(remove_noise(simgs[k]))
                    sfiles.append(name)
                pbar.update(1)

    images = np.array(images)
    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0], -1))
    sfiles = np.array(sfiles)

    return images, labels, sfiles


#
class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            "images.shape: %s labels.shape: %s" % (images.shape,
                                                   labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class DataSets(object):
    pass


# Onehot 编码转换回字符串
def onehot2number(label, LABELS):
    return LABELS[np.argmax(label)]


# 进行数据平衡
def balance(images, labels, sfiles, LABELS):
    a = []
    for i, label in enumerate(labels):
        label = onehot2number(label, LABELS)
        a.append([i, label])

    df = pd.DataFrame(a, columns=['i', 'label'])

    new_images = []
    new_labels = []
    new_sfiles = []
    for i in df['i']:
        new_images.append(images[i])
        new_labels.append(labels[i])
        new_sfiles.append(sfiles[i])
    images = np.array(new_images)
    labels = np.array(new_labels)
    sfiles = np.array(new_sfiles)

    return images, labels, sfiles


def make_data_sets(images, labels):
    trainX, testX, trainY, testY = model_selection.train_test_split(images, labels, test_size=0.20, random_state=42)

    data_sets = DataSets()
    data_sets.train = DataSet(trainX, trainY)
    data_sets.test = DataSet(testX, testY)

    return data_sets


def load(image_dir, IMAGE_H, IMAGE_W, LABEL_LENGTH, LABELS):
    images, labels, sfiles = load_data(image_dir, IMAGE_H, IMAGE_W, LABEL_LENGTH, LABELS)
    images, labels, sfiles = balance(images, labels, sfiles, LABELS)
    data_sets = make_data_sets(images, labels)
    return data_sets
