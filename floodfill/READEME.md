理论：

[OpenCV技巧 | 二值图孔洞填充方法与实现(附源码) - 腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1877771)

相关代码：

[【图像后处理】python+OpenCV填充孔洞 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/63919290)



思想：

找到背景所在的`连通域`，将其变为指定像素值。

故此需要找到对应的背景连通域所在的某个像素位置就行。

主流代码采用循环遍历的方式，但是可以对特定任务进行分析，如果某种任务的背景一定在某些位置，那么就可以特定的指定。

