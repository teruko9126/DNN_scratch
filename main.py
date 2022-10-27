# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from models import Relu, MRelu
import models.nn_parts as nn_parts

parser = argparse.ArgumentParser(description='deep learning training')
# 学習率の設定
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
# iteration回数の設定
parser.add_argument('--iter', default=10, type=float,
                    help='number of iteration')

# Relu関数を使うかMRelu関数を使うか選べる
parser.add_argument('--relu', action='store_true', help='use relu')
parser.add_argument('--mrelu', action='store_true', help='use mrelu')
# データセットをmnistかcircleから選ぶ
parser.add_argument('--mnist', action='store_true', help='dataset is mnist')
parser.add_argument('--circle', action='store_true', help='dataset is circle')

args = parser.parse_args()

# 学習データの選択
# 訓練データ数、テストデータ数、入力の画像を一列にしたときのサイズ、クラス数を定義しておく
# 初期値はMNISTに合わせている
train_data_num = 60000
test_data_num = 10000
input_data_size = 784
num_classes = 10
if args.mnist:
    # MNISTのデータを準備する
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # データの正規化(0~255 -> 0~1)
    train_examples = x_train.astype(np.float32) / 255
    train_labels = y_train.astype(np.float32)
    test_examples = x_test.astype(np.float32) / 255
    test_labels = y_test.astype(np.float32)

    # 訓練データのラベルをone-hotにする(学習のため)
    train_labels_onehot = tf.keras.utils.to_categorical(
        train_labels, num_classes=10).reshape([train_data_num, 10, 1])
elif args.circle:
    train_data_num = 60000
    test_data_num = 10000
    input_data_size = 2
    num_classes = 2

    # circleのデータを準備する
    (train_examples, train_labels) = nn_parts.circle_load_data(train_data_num)
    (test_examples, test_labels) = nn_parts.circle_load_data(test_data_num)

    # 訓練データのラベルをone-hotにする(学習のため)
    train_labels_onehot = tf.keras.utils.to_categorical(
        train_labels, num_classes=num_classes).reshape([train_data_num, 2, 1])

# 層の数の選択
layers_dims = [input_data_size, 50, 100, num_classes]


# ネットワークの選択
if args.relu:
    net = Relu.ReLU(layers_dims, args.lr)
elif args.mrelu:
    net = MRelu.MReLU(layers_dims, args.lr)

# 損失関数の定義
criterion = nn_parts.CrossEntropy()

# 訓練を行う
for i in range(train_data_num):
    out = net.forward(train_examples[i])
    criterion(out, train_labels_onehot[i])
    net.backward(criterion.backward())
    print(f'\r{i}', end='')
print("\n")

# テストを行う
point = 0
test_result = []
for i in range(test_data_num):
    ans = np.argmax(net.forward(test_examples[i]))
    if ans == test_labels[i]:
        point += 1
    print(f'\r{i}', end='')
    test_result.append(ans)
print("\n")

# circleデータのみ出力結果を図で確認
if args.circle:
    nn_parts.print_data_2d(test_examples, test_result)
# 正答率の出力
print(point/test_data_num)
