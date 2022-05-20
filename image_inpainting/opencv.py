# %%

import cv2
import numpy as np

img = cv2.imread('test.jpg')
print(img.shape)
print(img[:, 10, :])

# %%

import numpy as np
import cv2

# 读取图片
img = cv2.imread('test.jpg')
# 图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度二值化
_, mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 245, 255, cv2.THRESH_BINARY)

cv2.imshow('img_mask', mask)  # 掩模图
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

# INPAINT_TELEA算法
# cv2.INPAINT_TELEA （Fast Marching Method 快速行进算法）
# 对位于点附近、边界法线附近和边界轮廓上的像素赋予更多权重。一旦一个像素被修复，它将使用快速行进的方法移动到下一个最近的像素。
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)  # 10为领域大小
# src：输入8位1通道或3通道图像。
# inpaintMask：修复掩码，8位1通道图像。非零像素表示需要修复的区域。
# dst：输出与src具有相同大小和类型的图像。
# inpaintRadius：算法考虑的每个点的圆形邻域的半径。
# flags：
# INPAINT_NS基于Navier-Stokes的方法
# Alexandru Telea的INPAINT_TELEA方法
n = 230
m = 3
for i in range(100):
    print(i)
    # if i > 20000: m = 5
    # if i > 30000: m = 10
    # if i > 40000: m = 15
    dst = cv2.inpaint(dst, mask, m, cv2.INPAINT_NS)  # 10为领域大小
    # if i > 3000: n = 240
    # if i > 6000: n = 235
    # if i > 10000: n = 230
    # _,mask = cv2.threshold(cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY),n,255,cv2.THRESH_BINARY)
# INPAINT_NS算法
# cv2.INPAINT_NS（Fluid Dynamics Method 流体力学算法）
# 使用了流体力学的一些方法，基本原则是启发式的。首先沿着边从已知区域移动到未知区域（因为边是连续的）。
# 它在匹配修复区域边界处的渐变向量的同时，继续等高线（连接具有相同强度的点的线，就像等高线连接具有相同高程的点一样）。
# dst = cv2.inpaint(img,mask,30,cv2.INPAINT_NS) #10为领域大小
cv2.imshow('img_dst', dst)  # 掩模图
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

cv2.imwrite("test_dst.jpg", dst, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

# %%

import torch
from torch_pconv import PConv2d

import numpy as np
import cv2

# 读取图片
img = cv2.imread('test.jpg')
# 图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度二值化
_, mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 245, 255, cv2.THRESH_BINARY)
# [255,255,3] ==>[3,255,255]
# [c,h,w] ==> [n,c,h,w]
image = np.array(cv2.resize(img, (256, 256)), dtype=np.float32).transpose(2, 0, 1).reshape((1, 3, 256, 256)) / 255
mask = np.array(cv2.resize(mask, (256, 256)), dtype=np.float32).reshape((1, 256, 256)) / 255

print(image.shape)
print(mask.shape)

# print(torch.from_numpy(image))
# print(torch.from_numpy(mask))
pconv = PConv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=7,
    stride=1,
    padding=2,
    dilation=2,
    bias=True
)
# numpy ==> tensor
output, shrunk_masks = pconv(torch.from_numpy(image), torch.from_numpy(mask))

# %%

print(output.shape)
print(shrunk_masks.shape)

# %%

# import torch
# from torch_pconv import PConv2d
#
# import numpy as np
# import cv2
#
# # 读取图片
# img = cv2.imread('test.jpg')
# # 图像转换为灰度图像
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 灰度二值化
# _, mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 245, 255, cv2.THRESH_BINARY)
#
# image = np.array(cv2.resize(img, (256, 256)), dtype=np.float32).transpose(2, 0, 1).reshape((1, 3, 256, 256)) / 255
# mask = np.array(cv2.resize(mask, (256, 256)), dtype=np.float32).reshape((1, 256, 256)) / 255
#
# print(image.shape)
# print(mask.shape)
#
# # print(torch.from_numpy(image))
# # print(torch.from_numpy(mask))
# pconv = PConv2d(
#     in_channels=3,
#     out_channels=64,
#     kernel_size=7,
#     stride=1,
#     padding=2,
#     dilation=2,
#     bias=True
# )
#
# output, shrunk_masks = pconv(torch.from_numpy(image), torch.from_numpy(mask))
#
# # %%
#
# print(output.shape)
# print(shrunk_masks.shape)
