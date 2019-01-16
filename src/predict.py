#!/usr/bin/python
# -*- coding: UTF-8 -*-
import getopt
import os
import sys

import data
import resnet


def main(argv):
    image_dir = ''
    model_dir = ''
    try:
        opts, args = getopt.getopt(argv, "hi:m:", ["image=", "model="])
    except getopt.GetoptError:
        print('test.py -i <image_dir> -m <model_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <image_dir> -m <model_dir>')
            sys.exit()
        elif opt in ("-i", "--image"):
            image_dir = arg
        elif opt in ("-m", "--model"):
            model_dir = arg

    LABEL_LENGTH = 4  # 验证码字符数
    # LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    LABELS = "0123456789"  # 验证码字符组成

    IMAGE_H = 28  # 缩放后单个字符图片大小
    IMAGE_W = 28  # 缩放后单个字符图片大小
    IMAGE_C = 3  # 图片通道数

    model_file = model_dir + '/model.tfl'

    # 加载模型
    model = resnet.load(IMAGE_H, IMAGE_W, IMAGE_C, LABELS, model_file)

    # 预测
    def predict(filename):
        image = data.read_image(filename, IMAGE_H, IMAGE_W, LABEL_LENGTH)
        x_data = data.split_image(image, IMAGE_H, IMAGE_W, LABEL_LENGTH)
        y_preds = model.predict(x_data)

        label = ''
        for y_pred in y_preds:
            label = label + data.onehot2number(y_pred, LABELS)

        return label

    import random
    files = os.listdir(image_dir)

    for i in range(5):
        filename = os.path.join(image_dir, files[random.randint(0, len(files))])
        label = predict(filename)
        print(filename, label)


if __name__ == "__main__":
    main(sys.argv[1:])
