# -*- coding: utf-8 -*-
#
# import torch
# import numpy as np
# from torch import nn
# from PIL import Image
# from torch.autograd import Variable
# import torch.nn.functional as F
#
#
# def nn_conv2d(im):
#     # 用nn.Conv2d定义卷积操作
#     conv_op = nn.Conv2d(1, 1, 3, bias=False)
#     # 定义sobel算子参数
#     sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
#     # 将sobel算子转换为适配卷积操作的卷积核
#     sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
#     # 给卷积操作的卷积核赋值
#     conv_op.weight.data = torch.from_numpy(sobel_kernel)
#     # 对图像进行卷积操作
#     edge_detect = conv_op(Variable(im))
#     # 将输出转换为图片格式
#     edge_detect = edge_detect.squeeze().detach().numpy()
#     return edge_detect
#
#
# def functional_conv2d(im):
#     sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
#     sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
#     weight = Variable(torch.from_numpy(sobel_kernel))
#     edge_detect = F.conv2d(Variable(im), weight)
#     edge_detect = edge_detect.squeeze().detach().numpy()
#     return edge_detect
#
#
# def main():
#     # 读入一张图片，并转换为灰度图
#     im = Image.open('./input/demo2.jpg').convert('L')
#     # 将图片数据转换为矩阵
#     im = np.array(im, dtype='float32')
#     # 将图片矩阵转换为pytorch tensor,并适配卷积输入的要求
#     im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
#     # 边缘检测操作
#     # edge_detect = nn_conv2d(im)
#     edge_detect = functional_conv2d(im)
#     # 将array数据转换为image
#     im = Image.fromarray(edge_detect)
#     # image数据转换为灰度模式
#     im = im.convert('L')
#     # 保存图片
#     im.save('edge2.jpg', quality=95)
#
#
# if __name__ == "__main__":
#     main()

import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

def sobel_operator_test(image):
    # 定义Sobel算子的卷积核
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

    # 在图像上应用Sobel算子进行卷积操作
    gradient_x = F.conv2d(image, sobel_x.view(1, 1, 3, 3), padding=1)
    gradient_y = F.conv2d(image, sobel_y.view(1, 1, 3, 3), padding=1)

    # 计算梯度的幅度和角度
    gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
    gradient_angle = torch.atan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_angle

# 加载RGB图像
rgb_image_path = "./input/demo2.jpg"
# rgb_image = cv2.imread(rgb_image_path)
# rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

rgb_image_pil = Image.open(rgb_image_path)
rgb_image_tensor = ToTensor()(rgb_image_pil).unsqueeze(0)
# 1 1 480 640
grayscale_image = rgb_image_tensor.mean(dim=1, keepdim=True)

# 对灰度图像应用Sobel算子
grad_magnitude, grad_angle = sobel_operator_test(grayscale_image)

# 可视化结果
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(np.array(rgb_image_pil))
plt.title("Original RGB Image")

plt.subplot(1, 3, 2)
plt.imshow(grayscale_image.squeeze().numpy(), cmap='gray')
plt.title("Grayscale Image")

plt.subplot(1, 3, 3)
plt.imshow(grad_magnitude.squeeze().numpy(), cmap='gray')
plt.title("Sobel Gradient Magnitude")

plt.show()
# # 将RGB图像转换为灰度图像
# grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
# grayscale_image_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(grayscale_image, dtype=torch.float32)/255.0, 0), 0)

# 使用Canny算法进行边缘检测
# canny_edges = cv2.Canny(grayscale_image, 50, 150)  # 调整阈值根据需要

# # 对灰度图像应用Sobel算子
# grad_magnitude, grad_angle = sobel_operator(grayscale_image_tensor)
#
# # 可视化结果
# plt.figure(figsize=(12, 4))
#
# plt.subplot(1, 3, 1)
# plt.imshow(rgb_image)
# plt.title("Original RGB Image")
#
# plt.subplot(1, 3, 2)
# plt.imshow(grayscale_image, cmap='gray')
# plt.title("Grayscale Image")
#
# plt.subplot(1, 3, 3)
# plt.imshow(grad_magnitude.squeeze().numpy(), cmap='gray')
# plt.title("Gradient Magnitude")
#
# plt.show()
