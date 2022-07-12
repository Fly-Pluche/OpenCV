import cv2 as cv
import matplotlib.pyplot as plt
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

image0 = cv.imread(r'D:\workspace\PaddleRS\img\A\train_1.png') # 读取的是BGR
image=image0[:, :, ::-1]# 转换成RGB
reference0 = cv.imread(r'D:\workspace\PaddleRS\img\B\train_1.png') # 读取的是BGR
reference=reference0[:, :, [2, 1, 0] ]# 转换成RGB

matched = match_histograms(image, reference, multichannel=True) # 直方图匹配

# 显示图片
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

# 显示图片
ax1.imshow(image)
ax1.set_title('Spring')
ax2.imshow(reference)
ax2.set_title('Autumn')
ax3.imshow(matched)
ax3.set_title('Spring --> Autumn')

plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
plt.savefig('Spring_Autumn.jpg')
plt.show()


# 绘制直方图
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

for i, img in enumerate((image, reference, matched)):
    for c, c_color in enumerate(('red', 'green', 'blue')):
        img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
        axes[c, i].plot(bins, img_hist / img_hist.max())
        img_cdf, bins = exposure.cumulative_distribution(img[..., c])
        axes[c, i].plot(bins, img_cdf)
        axes[c, 0].set_ylabel(c_color)

# 设置标题
axes[0, 0].set_title('Source')
axes[0, 1].set_title('Reference')
axes[0, 2].set_title('Matched')

plt.tight_layout()
plt.savefig('Histogram.jpg')
plt.show()

