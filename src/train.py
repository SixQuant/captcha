#!/usr/bin/python
# -*- coding: UTF-8 -*-
import getopt
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

    IMAGE_H = 28  # 缩放后单个字符图片大小, 越小训练时间越短
    IMAGE_W = 28  # 缩放后单个字符图片大小, 越小训练时间越短
    IMAGE_C = 3  # 图片通道数

    model_file = model_dir + '/model.tfl'

    # 准备数据
    data_sets = data.load(image_dir, IMAGE_H, IMAGE_W, LABEL_LENGTH, LABELS)

    # 构建模型
    model = resnet.build(IMAGE_H, IMAGE_W, IMAGE_C, LABELS, model_file, learning_rate=0.01, val_acc_thresh=0.970)

    # 进行训练
    resnet.fit(model, data_sets, model_file, n_epoch=20)


if __name__ == "__main__":
    main(sys.argv[1:])
