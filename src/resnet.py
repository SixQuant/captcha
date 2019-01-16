# -*- coding: UTF-8 -*-

# 提前结束训练，用于保存最好的模型
import time

import tensorflow as tf
import tflearn


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        self.val_acc_thresh = val_acc_thresh

    def on_epoch_end(self, training_state):
        if training_state.val_acc >= self.val_acc_thresh and training_state.acc_value >= self.val_acc_thresh:
            raise StopIteration

    def on_train_end(self, training_state):
        self.val_acc = training_state.val_acc
        self.acc_value = training_state.acc_value
        print("Successfully left training! Final model accuracy:", training_state.acc_value)


def build(IMAGE_H, IMAGE_W, IMAGE_C, LABELS, model_file, learning_rate=0.01, val_acc_thresh=0.99):
    img_prep = tflearn.ImagePreprocessing()
    # img_prep.add_featurewise_zero_center(per_channel=True)  # 输入图像要减图像均值

    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_rotation(max_angle=10.0)  # 随机旋转角度
    # img_aug.add_random_blur(sigma_max=5.0)

    # Building Residual Network
    net = tflearn.input_data(shape=[None, IMAGE_H, IMAGE_W, IMAGE_C],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug,
                             name='input')

    net = tflearn.conv_2d(net, 16, 3,
                          regularizer='L2',
                          weights_init='variance_scaling',
                          weight_decay=0.0001,
                          name="conv1")  # 卷积处理, 16个卷积，卷积核大小为3，L2 正则化减少过拟合

    net = tflearn.residual_block(net, 1, 16, name="res1")  # 1 个残差层，输出16特征
    net = tflearn.residual_block(net, 1, 32, downsample=True, name="res2")  # 1 个残差层，输出32特征，降维1/2
    net = tflearn.residual_block(net, 1, 64, downsample=True, name="res3")  # 1 个残差层，输出64特征，降维1/2

    # Regression
    net = tflearn.fully_connected(net, len(LABELS), activation='softmax')
    mom = tflearn.Momentum(learning_rate, lr_decay=0.1, decay_step=32000, staircase=True)
    net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, max_checkpoints=1, tensorboard_verbose=3)

    return model


def load(IMAGE_H, IMAGE_W, IMAGE_C, LABELS, model_file):
    model = build(IMAGE_H, IMAGE_W, IMAGE_C, LABELS, model_file)
    # Load a model
    if tf.gfile.Exists(model_file + '.index'):
        model.load(model_file)
    return model


def fit(model, data_sets, model_file, n_epoch=20):
    start_time = time.time()
    fit_cb = EarlyStoppingCallback(val_acc_thresh=0.998)
    try:
        model.fit(data_sets.train.images,
                  data_sets.train.labels,
                  validation_set=(
                      data_sets.test.images,
                      data_sets.test.labels
                  ),
                  n_epoch=n_epoch,  # 完整数据集投喂次数，太多或太少会导致过拟合或欠拟合
                  batch_size=100,  # 每次训练获取的样本数
                  shuffle=True,  # 是否对数据进行洗牌
                  show_metric=True,  # 是否显示学习过程中的准确率
                  callbacks=fit_cb,  # 用于提前结束训练
                  run_id='vcode_resnet')


    except StopIteration as e:
        print("early stop")

    model.save(model_file)
    print('save trained model to ', model_file)

    duration = time.time() - start_time
    print('Training Duration %.3f sec' % (duration))
