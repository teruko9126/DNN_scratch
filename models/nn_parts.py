import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 各layerのparameterの初期化はHeの初期化
# https://paperswithcode.com/method/he-initialization

# circle : 円内のデータと円外のデータでラベルが異なるデータセット
# ラベルが1: データが中に存在している : [1,0]
# ラベルが1: データが外に存在している : [0,1]


def circle_load_data(num_data):
    data = np.random.rand(num_data, 2) * 2 - 1
    label = np.where(
        (data[0:, 0]) ** 2 + (data[0:, 1]) ** 2 >= (0.8) ** 2, 0, 1)

    return (data, label)

# circleの結果を可視化する


def print_data_2d(data, label):
    num_data = len(label)
    fig, ax1 = plt.subplots(figsize=(5, 5))
    colormap = np.array(['r', 'g'])
    ax1.scatter(data[0:, 0], data[0:, 1],
                c=colormap[label], s=0.1)

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()

# circleのデータを用いた際、中間層の出力がどのように変化するか可視化する


def print_data_3d(data, label, sum):
    num_data = len(label)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = np.array(['r', 'g'])

    ax.scatter(data[0:, 0:1], data[0:, 1:], sum,
               c=colormap[label[0:, 0:1].reshape(num_data)], s=0.1)

    plt.show()

# データの初期化を行う(heの初期値)
# https://paperswithcode.com/method/he-initialization


class Affine():
    def __init__(self, input_layer, output_layer, learning_rate):
        self.learning_rate = learning_rate
        self.weight = np.random.randn(
            output_layer, input_layer) * np.sqrt(2/input_layer)
        self.bias = np.zeros((output_layer, 1))

    def __call__(self, x):
        self.have_x = x
        return np.dot(self.weight, x) + self.bias

    def backward(self, chain_gradient):
        self.weight -= np.dot(chain_gradient,
                              self.have_x.transpose()) * self.learning_rate
        self.bias -= chain_gradient * self.learning_rate
        return np.dot(self.weight.transpose(), chain_gradient)


class ReLU():
    def __call__(self, x):
        self.have_x = x
        return np.where(x >= 0, x, 0)
      # parameters['relu_b1'] = np.zeros((layers_dims[1], 1))

    def backward(self, chain_gradient):
        return np.where(self.have_x >= 0, chain_gradient, 0)


class MReLU():
    def __call__(self, x):
        self.have_x = x
        relu_part = np.where(x >= 0, x, 0)
        mrelu_part = np.where(x <= 0, x, 0)
        return np.concatenate((relu_part, mrelu_part), axis=0)

    def backward(self, chain_gradient):
        relu_part_grad = np.where(
            self.have_x >= 0, chain_gradient[0:np.size(self.have_x)], 0)
        mrelu_part_grad = np.where(
            self.have_x <= 0, chain_gradient[np.size(self.have_x):], 0)
        return relu_part_grad + mrelu_part_grad


class Softmax():
    def __call__(self, x):
        self.have_x = x
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def backward(self, chain_gradient):
        p = self.__call__(self.have_x)
        return p * (1 - p) * chain_gradient


class CrossEntropy():

    def __call__(self, out, label):
        self.have_label = label
        self.have_out = out
        # Avoid division by zero
        out = np.clip(out, 1e-15, 1 - 1e-15)
        return - label * np.log(out) - (1 - label) * np.log(1 - out)

    def backward(self):
        # Avoid division by zero
        self.have_out = np.clip(self.have_out, 1e-15, 1 - 1e-15)
        return - (self.have_label / self.have_out) + (1 - self.have_label) / (1 - self.have_out)
