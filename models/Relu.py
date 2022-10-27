import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import models.nn_parts as nn_parts

# affine -> relu -> affine -> relu -> affine -> relu ->closs entropy


class ReLU:
    def __init__(self, layers_dims, learning_rate):
        self.affine1 = nn_parts.Affine(
            layers_dims[0], layers_dims[1], learning_rate)
        self.relu1 = nn_parts.ReLU()
        self.affine2 = nn_parts.Affine(
            layers_dims[1], layers_dims[2], learning_rate)
        self.relu2 = nn_parts.ReLU()
        self.affine3 = nn_parts.Affine(
            layers_dims[2], layers_dims[3], learning_rate)
        self.softmax = nn_parts.Softmax()

    def forward(self, data):
        out = data.flatten().reshape([-1, 1])
        out = self.affine1(out)
        out = self.relu1(out)
        out = self.affine2(out)
        out = self.relu2(out)
        out = self.affine3(out)
        out = self.softmax(out)
        return out

    def backward(self, loss_backward):
        chain_gradient = self.softmax.backward(loss_backward)
        chain_gradient = self.affine3.backward(chain_gradient)
        chain_gradient = self.relu2.backward(chain_gradient)
        chain_gradient = self.affine2.backward(chain_gradient)
        chain_gradient = self.relu1.backward(chain_gradient)
        chain_gradient = self.affine1.backward(chain_gradient)
