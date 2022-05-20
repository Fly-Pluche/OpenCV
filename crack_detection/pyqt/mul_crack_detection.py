import math
import numpy as np
from skimage import io
from skimage.morphology import medial_axis, skeletonize
from skimage import measure
from skimage import data
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import cv2
from numpy import *



def show_2dpoints(pointcluster, s=None, quivers=None, qscale=1):
    # pointcluster should be a list of numpy ndarray
    # This functions would show a list of pint cloud in different colors
    n = len(pointcluster)
    nmax = n
    if quivers is not None:
        nq = len(quivers)
        nmax = max(n, nq)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tomato', 'gold']
    if nmax < 10:
        colors = np.array(colors[0:nmax])
    else:
        colors = np.random.rand(nmax, 3)
    fig = plt.figure(num=1)
    ax = fig.add_subplot(1, 1, 1)

    if s is None:
        s = np.ones(n) * 2

    for i in range(n):
        ax.scatter(pointcluster[i][:, 0], pointcluster[i][:, 1], s=s[i], c=[colors[i]], alpha=0.6)

    if quivers is not None:
        for i in range(nq):
            ax.quiver(quivers[i][:, 0], quivers[i][:, 1], quivers[i][:, 2], quivers[i][:, 3], color=[colors[i]],
                      scale=qscale)

    plt.show()


def calcu_dis_from_ctrlpts(ctrlpts):
    if ctrlpts.shape[1] == 4:
        return np.sqrt(np.sum((ctrlpts[:, 0:2] - ctrlpts[:, 2:4]) ** 2, axis=1))
    else:
        return np.sqrt(np.sum((ctrlpts[:, [0, 2]] - ctrlpts[:, [3, 5]]) ** 2, axis=1))


def estimate_normals(points, n):
    """

    :rtype: object
    """
    pts = np.copy(points)
    # Leaf_size不会影响查询的结果，但可以显著影响查询速度和所需内存
    tree = KDTree(pts, leaf_size=2)
    # 返回离查询点最近的3个点的索引
    idx = tree.query(pts, k=n, return_distance=False, dualtree=False, breadth_first=False)

    # pts = np.concatenate((np.concatenate((pts[0].reshape(1,-1),pts),axis=0),pts[-1].reshape(1,-1)),axis=0)
    # 计算骨架线的法向量
    normals = []
    for i in range(0, pts.shape[0]):
        pts_for_normals = pts[idx[i, :], :]
        _, _, _, normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals


def SVD(points):
    # 二维，三维均适用
    # 二维直线，三维平面
    pts = points.copy()
    # 奇异值分解
    c = np.mean(pts, axis=0)
    A = pts - c  # shift the points
    A = A.T  # 3*n
    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)  # A=u*s*vh
    normal = u[:, -1]

    # 法向量归一化
    nlen = np.sqrt(np.dot(normal, normal))
    normal = normal / nlen
    # normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
    # u 每一列是一个方向
    # s 是对应的特征值
    # c >>> 点的中心
    # normal >>> 拟合的方向向量
    return u, s, c, normal


def estimate_normal_for_pos(pos, points, n):
    # estimate normal vectors at a given point
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(pos, k=n, return_distance=False, dualtree=False, breadth_first=False)
    # pts = np.concatenate((np.concatenate((pts[0].reshape(1,-1),pts),axis=0),pts[-1].reshape(1,-1)),axis=0)
    normals = []
    for i in range(0, pos.shape[0]):
        pts_for_normals = pts[idx[i, :], :]
        _, _, _, normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals


def get_crack_ctrlpts(centers, normals, bpoints, hband=5, vband=2):
    # main algorithm to obtain crack width
    cpoints = np.copy(centers)
    cnormals = np.copy(normals)
    # 基础坐标系
    xmatrix = np.array([[0, 1], [-1, 0]])
    # 坐标变换
    cnormalsx = np.dot(xmatrix, cnormals.T).T  # the normal of x axis
    N = cpoints.shape[0]

    interp_segm = []
    widths = []
    for i in range(N):
        try:
            # 遍历点
            ny = cnormals[i]
            nx = cnormalsx[i]
            tform = np.array([nx, ny])

            bpoints_loc = np.dot(tform, bpoints.T).T
            cpoints_loc = np.dot(tform, cpoints.T).T
            ci = cpoints_loc[i]

            ## 画图
            bl_ind = (bpoints_loc[:, 0] - (ci[0] - hband)) * (bpoints_loc[:, 0] - ci[0]) < 0
            br_ind = (bpoints_loc[:, 0] - ci[0]) * (bpoints_loc[:, 0] - (ci[0] + hband)) <= 0
            bl = bpoints_loc[bl_ind]  # left points
            br = bpoints_loc[br_ind]  # right points

            # 取中上点
            blt = bl[bl[:, 1] > np.mean(bl[:, 1])]
            # np.ptp最大值与最小值的差
            if np.ptp(blt[:, 1]) > vband:
                blt = blt[blt[:, 1] > np.mean(blt[:, 1])]
            # 取中下点
            blb = bl[bl[:, 1] < np.mean(bl[:, 1])]
            if np.ptp(blb[:, 1]) > vband:
                blb = blb[blb[:, 1] < np.mean(blb[:, 1])]

            brt = br[br[:, 1] > np.mean(br[:, 1])]
            if np.ptp(brt[:, 1]) > vband:
                brt = brt[brt[:, 1] > np.mean(brt[:, 1])]

            brb = br[br[:, 1] < np.mean(br[:, 1])]
            if np.ptp(brb[:, 1]) > vband:
                brb = brb[brb[:, 1] < np.mean(brb[:, 1])]

            # bh = np.vstack((bl,br))
            # bmax = np.max(bh[:,1])
            # bmin = np.min(bh[:,1])

            # blt = bl[bl[:,1]>bmax-vband] # left top points
            # blb = bl[bl[:,1]<bmin+vband] # left bottom points

            # brt = br[br[:,1]>bmax-vband] # right top points
            # brb = br[br[:,1]<bmin+vband] # right bottom points
            # 取极端值
            # np.argsort从小到大排的对应下标
            t1 = blt[np.argsort(blt[:, 0])[-1]]
            t2 = brt[np.argsort(brt[:, 0])[0]]

            b1 = blb[np.argsort(blb[:, 0])[-1]]
            b2 = brb[np.argsort(brb[:, 0])[0]]

            interp1 = (ci[0] - t1[0]) * ((t2[1] - t1[1]) / (t2[0] - t1[0])) + t1[1]
            interp2 = (ci[0] - b1[0]) * ((b2[1] - b1[1]) / (b2[0] - b1[0])) + b1[1]

            if interp1 - ci[1] > 0 and interp2 - ci[1] < 0:
                widths.append([i, interp1 - ci[1], interp2 - ci[1]])

            interps = np.array([[ci[0], interp1], [ci[0], interp2]])
            # 矩阵求逆  空间转化
            interps_rec = np.dot(np.linalg.inv(tform), interps.T).T

            # show_2dpoints([bpointsxl_loc1,bpointsxl_loc2,bpointsxr_loc1,bpointsxr_loc2,np.array([ptsl_1,ptsl_2]),np.array([ptsr_1,ptsr_2]),interps,ci.reshape(1,-1)],s=[1,1,1,1,20,20,20,20])
            interps_rec = interps_rec.reshape(1, -1)[0, :]
            interp_segm.append(interps_rec)
        except:
            print("the %d-th was wrong" % i)
            continue
    interp_segm = np.array(interp_segm)
    widths = np.array(widths)
    # check
    # show_2dpoints([np.array([[ci[0],interp1],[ci[0],interp2]]),np.array([t1,t2,b1,b2]),cpoints_loc,bl,br],[10,20,15,2,2])
    return interp_segm, widths



def detect(path1,path2):
    image1 = io.imread(path1, as_gray=True)
    image2 = io.imread(path2, as_gray=True)
    iw1, ih1 = image1.shape
    iw2, ih2 = image2.shape
    blobs1 = np.copy(image1)
    blobs2 = np.copy(image2)
    blobs1[blobs1 <= 0.5] = 0
    blobs1[blobs1 > 0.5] = 1
    blobs2[blobs2 <= 0.5] = 0
    blobs2[blobs2 > 0.5] = 1
    blobs1 = blobs1.astype(np.uint8)
    blobs2 = blobs2.astype(np.uint8)
    skeleton1 = skeletonize(blobs1)
    skeleton2 = skeletonize(blobs2)
    x1, y1 = np.where(skeleton1 > 0)
    x2, y2 = np.where(skeleton2 > 0)
    centers1 = np.hstack((x1.reshape(-1, 1), y1.reshape(-1, 1)))
    centers2 = np.hstack((x2.reshape(-1, 1), y2.reshape(-1, 1)))
    normals1 = estimate_normals(centers1, 3)
    normals2 = estimate_normals(centers2, 3)
    # search contours of the crack
    contours1 = measure.find_contours(blobs1, 0.8)
    contours2 = measure.find_contours(blobs2, 0.8)

    # 两条轮廓
    bl1 = contours1[0]
    br1 = contours1[1]
    bl2 = contours2[0]
    br2 = contours2[1]
    bpoints1 = np.vstack((bl1, br1))
    bpoints2 = np.vstack((bl2, br2))

    # TODO:iw ,ih
    bpixel1 = np.zeros((iw1, ih1, 3), dtype=np.uint8)
    bpixel2 = np.zeros((iw2, ih2, 3), dtype=np.uint8)
    bpoints1 = bpoints1.astype(np.int)
    bpoints2 = bpoints2.astype(np.int)
    bpixel1[bpoints1[:, 0], bpoints1[:, 1], 0] = 255
    bpixel2[bpoints2[:, 0], bpoints2[:, 1], 0] = 255
    skeleton_pixel1 = np.zeros((iw1, ih1, 3), dtype=np.uint8)
    skeleton_pixel2 = np.zeros((iw2, ih2, 3), dtype=np.uint8)
    skeleton_pixel1[skeleton1, 1] = 255
    skeleton_pixel2[skeleton2, 1] = 255
    bpixel_and_skeleton1 = np.copy(bpixel1)
    bpixel_and_skeleton2 = np.copy(bpixel2)
    bpixel_and_skeleton1[skeleton1, 1] = 255
    bpixel_and_skeleton2[skeleton2, 1] = 255

    fig, ax = plt.subplots()
    ########################################特征点##################################
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread('../cracks/yuantu/7.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../cracks/yuantu/8.png', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('../cracks/gujia/7.2.png', cv2.IMREAD_GRAYSCALE)#3,4
    img4 = cv2.imread('../cracks/gujia/8.2.png', cv2.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    coordinates_1 = []
    coordinates_2 = []
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.7 * m2.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
            matchesMask[i] = [1, 0]
            pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
            coordinates_1.append(pt1)
            coordinates_2.append(pt2)
            if i % 5 == 0:
                cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 255), -1)
                cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 5, (255, 0, 255), -1)
    ps = []
    for pc in coordinates_1:
        x_1 = int(pc[0])
        y_1 = int(pc[1])
        # print(x,y)
        iw1, ih1 = img3.shape
        #y不变，变x
        for dx1 in range(0,iw1,1):
            c = img3[dx1,y_1]
            if c > 20:
                ps.append((dx1,y_1))
    # tm1 = img3_1.copy()
    # for p in ps:
    #     cv2.circle(tm1,p,1,(0,255,255))
    #cv2.imshow("result",tm1)

    pr = []
    for pd in coordinates_2:
        x_2 = int(pd[0])
        y_2 = int(pd[1])
        iw2, ih2 = img4.shape
        for dx2 in range(0,iw2,1):
            c = img3[dx2,y_2]
            if c > 20:
                pr.append((dx2,y_2))
    #######################################计算宽度############

    distance_1 = []
    distance_2 = []
    for i in range(0, len(ps), 1):
        pos1 = np.array(ps[i]).reshape(1, -1)
        posn1 = estimate_normal_for_pos(pos1, centers1, 3)
        interps1, widths21 = get_crack_ctrlpts(pos1, posn1, bpoints1, hband=2, vband=2)
        if len(interps1) == 0 or len(widths21) == 0:
            continue
        else:
            distance1 = math.sqrt(pow(interps1[0, 0] - interps1[0, 1], 2) + pow(interps1[0, 2] - interps1[0, 3], 2))
            #print('distance', distance)
            distance_1.append(distance1)
            for i in range(interps1.shape[0]):
                ax.plot([interps1[i, 1], interps1[i, 3]], [interps1[i, 0], interps1[i, 2]], c='c', ls='-', lw=5, marker='o',
                        ms=8,
                        mec='c', mfc='c')
        ax.imshow(bpixel_and_skeleton1)
        ax.axis('off')

    for i in range(0, len(pr), 1):
        pos2 = np.array(pr[i]).reshape(1, -1)
        posn2 = estimate_normal_for_pos(pos2, centers2, 3)
        interps2, widths22 = get_crack_ctrlpts(pos2, posn2, bpoints2, hband=2, vband=2)
        if len(interps2) == 0 or len(widths22) == 0:
            continue
        else:
            distance2 = math.sqrt(pow(interps2[0, 0] - interps2[0, 1], 2) + pow(interps2[0, 2] - interps2[0, 3], 2))
            # print('distance', distance)
            distance_2.append(distance2)
            for i in range(interps2.shape[0]):
                ax.plot([interps2[i, 1], interps2[i, 3]], [interps2[i, 0], interps2[i, 2]], c='c', ls='-', lw=5,marker='o',
                        ms=8,
                        mec='c', mfc='c')
        ax.imshow(bpixel_and_skeleton2)
    ax.axis('off')
    # print('distance_1',distance_1)
    # print('distance_2',distance_2)
    distance_1_mean = mean(distance_1)
    distance_2_mean = mean(distance_2)
    #裂缝一的平均宽度
    # print('distance_1_mean',distance_1_mean)
    #裂缝2的平均宽度
    # print('distance_2_mean',distance_2_mean)
    #裂缝一的最大宽度和裂缝2的最大宽度
    # print('max(distance_1),max(distance_2)',max(distance_1),max(distance_2))
    return round(distance_1_mean,2),round(distance_2_mean,2),round(max(distance_1),2),round(max(distance_2),2)


if __name__ == '__main__':
    path1 = "./cracks/erzhi/7.png"
    path2 = "./cracks/erzhi/8.png"
    detect(path1,path2)