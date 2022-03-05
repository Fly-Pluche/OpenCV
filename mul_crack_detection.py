import math
import numpy as np
from skimage import io
from skimage.morphology import medial_axis, skeletonize
from skimage import measure
from skimage import data
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


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


path = "./image/3.jpg"

image = io.imread(path, as_gray=True)
iw, ih = image.shape
import  cv2

image=cv2.imread(path)[:,:,0]
blobs = np.copy(image)
blobs[blobs < 128] = 0
blobs[blobs > 128] = 1

blobs = blobs.astype(np.uint8)

# Compare with other skeletonization algorithms
skeleton = skeletonize(blobs)
# skeleton_lee = skeletonize(blobs, method='lee')
x, y = np.where(skeleton > 0)
centers = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
normals = estimate_normals(centers, 3)

# search contours of the crack
contours = measure.find_contours(blobs, 0.8)

# 两条轮廓
bl = contours[0]
br = contours[1]

bpoints = np.vstack((bl, br))

# interp_segm, widths = get_crack_ctrlpts(centers,normals,bpoints,hband=2,vband=2)

# TODO:iw ,ih
bpixel = np.zeros((iw, ih, 3), dtype=np.uint8)
bpoints = bpoints.astype(np.int)
bpixel[bpoints[:, 0], bpoints[:, 1], 0] = 255

skeleton_pixel = np.zeros((iw, ih, 3), dtype=np.uint8)
skeleton_pixel[skeleton, 1] = 255

bpixel_and_skeleton = np.copy(bpixel)
bpixel_and_skeleton[skeleton, 1] = 255

fig, ax = plt.subplots()
for i in range(0,len(centers),10):
    pos = np.array(centers[i]).reshape(1, -1)

    posn = estimate_normal_for_pos(pos, centers, 3)

    interps, widths2 = get_crack_ctrlpts(pos, posn, bpoints, hband=1.5, vband=2)
    if len(interps)==0 or len(widths2) ==0:
        continue
    else:
        print('distance',math.sqrt(pow(interps[0,0]-interps[0,1],2)+pow(interps[0,2]-interps[0,3],2)))
        ax.imshow(bpixel_and_skeleton)

        for i in range(interps.shape[0]):
            ax.plot([interps[i, 1], interps[i, 3]], [interps[i, 0], interps[i, 2]], c='c', ls='-', lw=5, marker='o', ms=8,
                       mec='c', mfc='c')

    # ax[2].set_title('skeletonize')
ax.axis('off')

fig.tight_layout()
plt.savefig('1.png')
plt.show()
