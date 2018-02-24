# coding: utf-8

import math
import numpy as np
import enum


class ConV2D(object):

    def __init__(self, shape, output_channels, k_size=3, stride=1, method='VALID'):
        """
        根据上面的公式，我们知道实现一个卷积前向计算的操作，我们需要知道以下信息：

        输入数据的shape = [N,W,H,C] N=Batchsize/ W=width/ H=height/ C=channels
        卷积核的尺寸ksize ,个数output_channels, kernel shape [output_channels,k,k,C]
        卷积的步长，基本默认为1.
        卷积的方法，VALID or SAME，即是否通过padding保持输出图像与输入图像的大小不变
        :param shape:
        :param output_channels:
        :param k_size:
        :param stride:
        :param method:
        """
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.batch_size = shape[0]
        self.stride = stride
        self.k_size = k_size
        self.method = method

        """初始化一些参数"""
        weights_scale = math.sqrt(k_size * k_size * self.input_channels / 2)
        self.weights = np.random.standard_normal(
            (k_size, k_size, self.input_channels, self.output_channels)
        ) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale

        if method == 'VALID':
            self.eta = np.zeros(
                (
                    shape[0],
                    (shape[1] - k_size + 1) / self.stride,
                    (shape[1] - k_size + 1) / self.stride,
                    self.output_channels
                 )
            )

        if method == 'SAME':
            self.eta = np.zeros(
                (
                    shape[0],
                    shape[1] / self.stride,
                    shape[2] / self.stride,
                    self.output_channels
                )
            )

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

        self.col_image_i = None
        self.col_image = None

        if (shape[1] - k_size) % stride != 0:
            print('input tensor width can\'t fit stride')
        if (shape[2] - k_size) % stride != 0:
            print('input tensor height can\'t fit stride')

    def forward(self, x):
        """
        向前传播
        :param x:
        :return:
        """
        col_weights = self.weights.reshape([-1, self.output_channels])
        if self.method == 'SAME':
            x = np.pad(x, (
                (0, 0), (self.k_size / 2, self.k_size / 2), (self.k_size / 2, self.k_size / 2), (0, 0)),
                             'constant', constant_values=0)

        self.col_image = []
        con_v_out = np.zeros(self.eta.shape)
        for i in range(self.batch_size):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.k_size, self.stride)
            con_v_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return con_v_out


def im2col(image, k_size, stride):
    # image is a 4d tensor([batch_size, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - k_size + 1, stride):
        for j in range(0, image.shape[2] - k_size + 1, stride):
            col = image[:, i:i + k_size, j:j + k_size, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col


if __name__ == "__main__":
    # 标准正态分布
    a = np.random.standard_normal((2, 3))
    print(a)
    print(np.sum(a * a))
    print(math.sqrt(9))
    pass
