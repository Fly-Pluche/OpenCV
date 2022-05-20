#!/usr/bin/env python
# coding: utf-8

# In[2]:


#predict
import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

SZ = 20          #训练图片长宽
MAX_WIDTH = 1000 #原始图片最大宽度
Min_Area = 2000  #车牌区域允许最大面积
PROVINCE_START = 1000
#读取图片文件
def imreadex(filename):
	return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
	
def point_limit(point):
	if point[0] < 0:
		point[0] = 0
	if point[1] < 0:
		point[1] = 0

#根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
	up_point = -1#上升点
	is_peak = False
	if histogram[0] > threshold:
		up_point = 0
		is_peak = True
	wave_peaks = []
	for i,x in enumerate(histogram):
		if is_peak and x < threshold:
			if i - up_point > 2:
				is_peak = False
				wave_peaks.append((up_point, i))
		elif not is_peak and x >= threshold:
			is_peak = True
			up_point = i
	if is_peak and up_point != -1 and i - up_point > 4:
		wave_peaks.append((up_point, i))
	return wave_peaks

#根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
	part_cards = []
	for wave in waves:
		part_cards.append(img[:, wave[0]:wave[1]])
	return part_cards

#来自opencv的sample，用于svm训练
def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img
#来自opencv的sample，用于svm训练
def preprocess_hog(digits):
	samples = []
	for img in digits:
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bin_n = 16
		bin = np.int32(bin_n*ang/(2*np.pi))
		bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)
		
		# transform to Hellinger kernel
		eps = 1e-7
		hist /= hist.sum() + eps
		hist = np.sqrt(hist)
		hist /= norm(hist) + eps
		
		samples.append(hist)
	return np.float32(samples)
#不能保证包括所有省份
provinces = [
"zh_cuan", "川",
"zh_e", "鄂",
"zh_gan", "赣",
"zh_gan1", "甘",
"zh_gui", "贵",
"zh_gui1", "桂",
"zh_hei", "黑",
"zh_hu", "沪",
"zh_ji", "冀",
"zh_jin", "津",
"zh_jing", "京",
"zh_jl", "吉",
"zh_liao", "辽",
"zh_lu", "鲁",
"zh_meng", "蒙",
"zh_min", "闽",
"zh_ning", "宁",
"zh_qing", "靑",
"zh_qiong", "琼",
"zh_shan", "陕",
"zh_su", "苏",
"zh_sx", "晋",
"zh_wan", "皖",
"zh_xiang", "湘",
"zh_xin", "新",
"zh_yu", "豫",
"zh_yu1", "渝",
"zh_yue", "粤",
"zh_yun", "云",
"zh_zang", "藏",
"zh_zhe", "浙"
]
class StatModel(object):
	def load(self, fn):
		self.model = self.model.load(fn)  
	def save(self, fn):
		self.model.save(fn)
class SVM(StatModel):
	def __init__(self, C = 1, gamma = 0.5):
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(C)
		self.model.setKernel(cv2.ml.SVM_RBF)
		self.model.setType(cv2.ml.SVM_C_SVC)
#训练svm
	def train(self, samples, responses):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
#字符识别
	def predict(self, samples):
		r = self.model.predict(samples)
		return r[1].ravel()

class CardPredictor:
	def __init__(self):
		#车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
		pass
		# f = open('config.js')
		# j = json.load(f)
		# for c in j["config"]:
		# 	if c["open"]:
		# 		self.cfg = c.copy()
		# 		break
		# else:
		# 	raise RuntimeError('没有设置有效配置参数')

	def __del__(self):
		self.save_traindata()
	def train_svm(self):
		#识别英文字母和数字
		self.model = SVM(C=1, gamma=0.5)
		#识别中文
		self.modelchinese = SVM(C=1, gamma=0.5)
		if os.path.exists("svm.dat"):
			self.model.load("svm.dat")
		else:
			chars_train = []
			chars_label = []
			
			for root, dirs, files in os.walk("train\\chars2"):
				if len(os.path.basename(root)) > 1:
					continue
				root_int = ord(os.path.basename(root))
				for filename in files:
					filepath = os.path.join(root,filename)
					digit_img = cv2.imread(filepath)
					digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
					chars_train.append(digit_img)
					#chars_label.append(1)
					chars_label.append(root_int)
			
			chars_train = list(map(deskew, chars_train))
			chars_train = preprocess_hog(chars_train)
			#chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
			chars_label = np.array(chars_label)
			self.model.train(chars_train, chars_label)
		if os.path.exists("svmchinese.dat"):
			self.modelchinese.load("svmchinese.dat")
		else:
			chars_train = []
			chars_label = []
			for root, dirs, files in os.walk("train\\charsChinese"):
				if not os.path.basename(root).startswith("zh_"):
					continue
				pinyin = os.path.basename(root)
				index = provinces.index(pinyin) + PROVINCE_START + 1 #1是拼音对应的汉字
				for filename in files:
					filepath = os.path.join(root,filename)
					digit_img = cv2.imread(filepath)
					digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
					chars_train.append(digit_img)
					#chars_label.append(1)
					chars_label.append(index)
			chars_train = list(map(deskew, chars_train))
			chars_train = preprocess_hog(chars_train)
			#chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
			chars_label = np.array(chars_label)
			print(chars_train.shape)
			self.modelchinese.train(chars_train, chars_label)

	def save_traindata(self):
		pass
		# if not os.path.exists("svm.dat"):
		# 	self.model.save("svm.dat")
		# if not os.path.exists("svmchinese.dat"):
		# 	self.modelchinese.save("svmchinese.dat")

	def accurate_place(self, card_img_hsv, limit1, limit2, color):
		row_num, col_num = card_img_hsv.shape[:2]
		xl = col_num
		xr = 0
		yh = 0
		yl = row_num
		#col_num_limit = self.cfg["col_num_limit"]
		# row_num_limit = self.cfg["row_num_limit"]
		row_num_limit = 10

		col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5#绿色有渐变
		for i in range(row_num):
			count = 0
			for j in range(col_num):
				H = card_img_hsv.item(i, j, 0)
				S = card_img_hsv.item(i, j, 1)
				V = card_img_hsv.item(i, j, 2)
				if limit1 < H <= limit2 and 34 < S and 46 < V:
					count += 1
			if count > col_num_limit:
				if yl > i:
					yl = i
				if yh < i:
					yh = i
		for j in range(col_num):
			count = 0
			for i in range(row_num):
				H = card_img_hsv.item(i, j, 0)
				S = card_img_hsv.item(i, j, 1)
				V = card_img_hsv.item(i, j, 2)
				if limit1 < H <= limit2 and 34 < S and 46 < V:
					count += 1
			if count > row_num - row_num_limit:
				if xl > j:
					xl = j
				if xr < j:
					xr = j
		return xl, xr, yh, yl
		
	def predict(self, car_pic, resize_rate=1):
		if type(car_pic) == type(""):
			img = imreadex(car_pic)
		else:
			img = car_pic
		pic_hight, pic_width = img.shape[:2]
		if pic_width > MAX_WIDTH:
			pic_rate = MAX_WIDTH / pic_width
			img = cv2.resize(img, (MAX_WIDTH, int(pic_hight*pic_rate)), interpolation=cv2.INTER_LANCZOS4)
		
		if resize_rate != 1:
			img = cv2.resize(img, (int(pic_width*resize_rate), int(pic_hight*resize_rate)), interpolation=cv2.INTER_LANCZOS4)
			pic_hight, pic_width = img.shape[:2]
			
		print("h,w:", pic_hight, pic_width)
		# blur = self.cfg["blur"]
		blur = 3
		#高斯去噪
		if blur > 0:
			img = cv2.GaussianBlur(img, (blur, blur), 0)#图片分辨率调整
		oldimg = img
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#equ = cv2.equalizeHist(img)
		#img = np.hstack((img, equ))
		#去掉图像中不会是车牌的区域
		kernel = np.ones((20, 20), np.uint8)
		img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
		img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0);

		#找到图像边缘
		ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		img_edge = cv2.Canny(img_thresh, 100, 200)
		#使用开运算和闭运算让图像边缘成为一个整体
		# kernel = np.ones((self.cfg["morphologyr"], self.cfg["morphologyc"]), np.uint8)
		kernel = np.ones((15, 15), np.uint8)
		img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
		img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

		#查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
		try:
			contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		except ValueError:
			image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
		print('len(contours)', len(contours))
		#一一排除不是车牌的矩形区域
		car_contours = []
		for cnt in contours:
			rect = cv2.minAreaRect(cnt)
			area_width, area_height = rect[1]
			if area_width < area_height:
				area_width, area_height = area_height, area_width
			wh_ratio = area_width / area_height
			#print(wh_ratio)
			#要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
			if wh_ratio > 2 and wh_ratio < 5.5:
				car_contours.append(rect)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				#oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
				#cv2.imshow("edge4", oldimg)
				#cv2.waitKey(0)

		print(len(car_contours))

		print("精确定位")
		card_imgs = []
		#矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
		for rect in car_contours:
			if rect[2] > -1 and rect[2] < 1:#创造角度，使得左、高、右、低拿到正确的值
				angle = 1
			else:
				angle = rect[2]
			rect = (rect[0], (rect[1][0]+5, rect[1][1]+5), angle)#扩大范围，避免车牌边缘被排除

			box = cv2.boxPoints(rect)
			heigth_point = right_point = [0, 0]
			left_point = low_point = [pic_width, pic_hight]
			for point in box:
				if left_point[0] > point[0]:
					left_point = point
				if low_point[1] > point[1]:
					low_point = point
				if heigth_point[1] < point[1]:
					heigth_point = point
				if right_point[0] < point[0]:
					right_point = point

			if left_point[1] <= right_point[1]:#正角度
				new_right_point = [right_point[0], heigth_point[1]]
				pts2 = np.float32([left_point, heigth_point, new_right_point])#字符只是高度需要改变
				pts1 = np.float32([left_point, heigth_point, right_point])
				M = cv2.getAffineTransform(pts1, pts2)
				dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
				point_limit(new_right_point)
				point_limit(heigth_point)
				point_limit(left_point)
				card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
				card_imgs.append(card_img)
				#cv2.imshow("card", card_img)
				#cv2.waitKey(0)
			elif left_point[1] > right_point[1]:#负角度
				
				new_left_point = [left_point[0], heigth_point[1]]
				pts2 = np.float32([new_left_point, heigth_point, right_point])#字符只是高度需要改变
				pts1 = np.float32([left_point, heigth_point, right_point])
				M = cv2.getAffineTransform(pts1, pts2)
				dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
				point_limit(right_point)
				point_limit(heigth_point)
				point_limit(new_left_point)
				card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
				card_imgs.append(card_img)
				#cv2.imshow("card", card_img)
				#cv2.waitKey(0)
		#开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
		colors = []
		for card_index,card_img in enumerate(card_imgs):
			green = yello = blue = black = white = 0
			card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
			#有转换失败的可能，原因来自于上面矫正矩形出错
			if card_img_hsv is None:
				continue
			row_num, col_num= card_img_hsv.shape[:2]
			card_img_count = row_num * col_num

			for i in range(row_num):
				for j in range(col_num):
					H = card_img_hsv.item(i, j, 0)
					S = card_img_hsv.item(i, j, 1)
					V = card_img_hsv.item(i, j, 2)
					if 11 < H <= 34 and S > 34:#图片分辨率调整
						yello += 1
					elif 35 < H <= 99 and S > 34:#图片分辨率调整
						green += 1
					elif 99 < H <= 124 and S > 34:#图片分辨率调整
						blue += 1
					
					if 0 < H <180 and 0 < S < 255 and 0 < V < 46:
						black += 1
					elif 0 < H <180 and 0 < S < 43 and 221 < V < 225:
						white += 1
			color = "no"

			limit1 = limit2 = 0
			if yello*2 >= card_img_count:
				color = "yello"
				limit1 = 11
				limit2 = 34#有的图片有色偏偏绿
			elif green*2 >= card_img_count:
				color = "green"
				limit1 = 35
				limit2 = 99
			elif blue*2 >= card_img_count:
				color = "blue"
				limit1 = 100
				limit2 = 124#有的图片有色偏偏紫
			elif black + white >= card_img_count*0.7:#TODO
				color = "bw"
			print(color)
			colors.append(color)
			print(blue, green, yello, black, white, card_img_count)
			#cv2.imshow("color", card_img)
			#cv2.waitKey(0)
			if limit1 == 0:
				continue
			#以上为确定车牌颜色
			#以下为根据车牌颜色再定位，缩小边缘非车牌边界
			xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
			if yl == yh and xl == xr:
				continue
			need_accurate = False
			if yl >= yh:
				yl = 0
				yh = row_num
				need_accurate = True
			if xl >= xr:
				xl = 0
				xr = col_num
				need_accurate = True
			card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh-yl)//4 else card_img[yl-(yh-yl)//4:yh, xl:xr]
			if need_accurate:#可能x或y方向未缩小，需要再试一次
				card_img = card_imgs[card_index]
				card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
				xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
				if yl == yh and xl == xr:
					continue
				if yl >= yh:
					yl = 0
					yh = row_num
				if xl >= xr:
					xl = 0
					xr = col_num
			card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh-yl)//4 else card_img[yl-(yh-yl)//4:yh, xl:xr]
		#以上为车牌定位
		#以下为识别车牌中的字符
		predict_result = []
		roi = None
		card_color = None
		for i, color in enumerate(colors):
			if color in ("blue", "yello", "green"):
				card_img = card_imgs[i]
				gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
				#黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
				if color == "green" or color == "yello":
					gray_img = cv2.bitwise_not(gray_img)
				ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
				#查找水平直方图波峰
				x_histogram  = np.sum(gray_img, axis=1)
				x_min = np.min(x_histogram)
				x_average = np.sum(x_histogram)/x_histogram.shape[0]
				x_threshold = (x_min + x_average)/2
				wave_peaks = find_waves(x_threshold, x_histogram)
				if len(wave_peaks) == 0:
					print("peak less 0:")
					continue
				#认为水平方向，最大的波峰为车牌区域
				wave = max(wave_peaks, key=lambda x:x[1]-x[0])
				gray_img = gray_img[wave[0]:wave[1]]
				#查找垂直直方图波峰
				row_num, col_num= gray_img.shape[:2]
				#去掉车牌上下边缘1个像素，避免白边影响阈值判断
				gray_img = gray_img[1:row_num-1]
				y_histogram = np.sum(gray_img, axis=0)
				y_min = np.min(y_histogram)
				y_average = np.sum(y_histogram)/y_histogram.shape[0]
				y_threshold = (y_min + y_average)/5#U和0要求阈值偏小，否则U和0会被分成两半

				wave_peaks = find_waves(y_threshold, y_histogram)

				#for wave in wave_peaks:
				#	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2) 
				#车牌字符数应大于6
				if len(wave_peaks) <= 6:
					print("peak less 1:", len(wave_peaks))
					continue
				
				wave = max(wave_peaks, key=lambda x:x[1]-x[0])
				max_wave_dis = wave[1] - wave[0]
				#判断是否是左侧车牌边缘
				if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis/3 and wave_peaks[0][0] == 0:
					wave_peaks.pop(0)
				
				#组合分离汉字
				cur_dis = 0
				for i,wave in enumerate(wave_peaks):
					if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
						break
					else:
						cur_dis += wave[1] - wave[0]
				if i > 0:
					wave = (wave_peaks[0][0], wave_peaks[i][1])
					wave_peaks = wave_peaks[i+1:]
					wave_peaks.insert(0, wave)
				
				#去除车牌上的分隔点
				point = wave_peaks[2]
				if point[1] - point[0] < max_wave_dis/3:
					point_img = gray_img[:,point[0]:point[1]]
					if np.mean(point_img) < 255/5:
						wave_peaks.pop(2)
				
				if len(wave_peaks) <= 6:
					print("peak less 2:", len(wave_peaks))
					continue
				part_cards = seperate_card(gray_img, wave_peaks)
				for i, part_card in enumerate(part_cards):
					#可能是固定车牌的铆钉
					if np.mean(part_card) < 255/5:
						print("a point")
						continue
					part_card_old = part_card
					#w = abs(part_card.shape[1] - SZ)//2
					w = part_card.shape[1] // 3
					part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])
					part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
					#cv2.imshow("part", part_card_old)
					#cv2.waitKey(0)
					#cv2.imwrite("u.jpg", part_card)
					#part_card = deskew(part_card)
					part_card = preprocess_hog([part_card])
					if i == 0:
						resp = self.modelchinese.predict(part_card)
						charactor = provinces[int(resp[0]) - PROVINCE_START]
					else:
						resp = self.model.predict(part_card)
						charactor = chr(resp[0])
					#判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
					if charactor == "1" and i == len(part_cards)-1:
						if part_card_old.shape[0]/part_card_old.shape[1] >= 8:#1太细，认为是边缘
							print(part_card_old.shape)
							continue
					predict_result.append(charactor)
				roi = card_img
				card_color = color
				break
		
		return predict_result, roi, card_color#识别到的字符、定位的车牌图像、车牌颜色

if __name__ == '__main__':
	c = CardPredictor()
	c.train_svm()
	r, roi, color = c.predict("2.jpg")
	print(r)


# In[2]:


#surface
import tkinter as tk
from tkinter.constants import BOTH
from tkinter.filedialog import *
from tkinter import ttk
import predict
import cv2
from PIL import Image, ImageTk
import threading
import time


class Surface(ttk.Frame):
	pic_path = ""
	viewhigh = 600
	viewwide = 600
	update_time = 0
	thread = None
	thread_run = False
	camera = None
	color_transform = {"green":("绿牌","#55FF55"), "yello":("黄牌","#FFFF00"), "blue":("蓝牌","#6666FF")}
		
	def __init__(self, win):
		ttk.Frame.__init__(self, win)
		frame_left = ttk.Frame(self)
		frame_right1 = ttk.Frame(self)
		frame_right2 = ttk.Frame(self)
		win.title("车牌车型识别(山大威海)")
		win.state("zoomed")
		self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
		frame_left.pack(side="left",expand=1,fill=BOTH)
		frame_right1.pack(side="top",expand=1,fill=tk.Y)
		frame_right2.pack(side="right",expand=0)
		ttk.Label(frame_left, text='原图：').pack(anchor="nw") 
		ttk.Label(frame_right1, text='车牌位置：').grid(column=0, row=0, sticky=tk.W)
		
		from_pic_ctl = ttk.Button(frame_right2, text="来自图片", width=20, command=self.from_pic)
		from_vedio_ctl = ttk.Button(frame_right2, text="来自摄像头", width=20, command=self.from_vedio)
		self.image_ctl = ttk.Label(frame_left)
		self.image_ctl.pack(anchor="nw")
		
		self.roi_ctl = ttk.Label(frame_right1)
		self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
		ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
		self.r_ctl = ttk.Label(frame_right1, text="")
		self.r_ctl.grid(column=0, row=3, sticky=tk.W)
		self.color_ctl = ttk.Label(frame_right1, text="", width="20")
		self.color_ctl.grid(column=0, row=4, sticky=tk.W)
		from_vedio_ctl.pack(anchor="se", pady="5")
		from_pic_ctl.pack(anchor="se", pady="5")
		self.predictor = predict.CardPredictor()
		self.predictor.train_svm()
		
	def get_imgtk(self, img_bgr):
		img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		im = Image.fromarray(img)
		imgtk = ImageTk.PhotoImage(image=im)
		wide = imgtk.width()
		high = imgtk.height()
		if wide > self.viewwide or high > self.viewhigh:
			wide_factor = self.viewwide / wide
			high_factor = self.viewhigh / high
			factor = min(wide_factor, high_factor)
			
			wide = int(wide * factor)
			if wide <= 0 : wide = 1
			high = int(high * factor)
			if high <= 0 : high = 1
			im=im.resize((wide, high), Image.ANTIALIAS)
			imgtk = ImageTk.PhotoImage(image=im)
		return imgtk
	
	def show_roi(self, r, roi, color):
		if r :
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
			roi = Image.fromarray(roi)
			self.imgtk_roi = ImageTk.PhotoImage(image=roi)
			self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
			self.r_ctl.configure(text=str(r))
			self.update_time = time.time()
			try:
				c = self.color_transform[color]
				self.color_ctl.configure(text=c[0], background=c[0], state='enable')
			except: 
				self.color_ctl.configure(state='disabled')
		elif self.update_time + 8 < time.time():
			self.roi_ctl.configure(state='disabled')
			self.r_ctl.configure(text="")
			self.color_ctl.configure(state='disabled')
		
	def from_vedio(self):
		if self.thread_run:
			return
		if self.camera is None:
			self.camera = cv2.VideoCapture(0)
			if not self.camera.isOpened():
				mBox.showwarning('警告', '摄像头打开失败！')
				self.camera = None
				return
		self.thread = threading.Thread(target=self.vedio_thread, args=(self,))
		self.thread.setDaemon(True)
		self.thread.start()
		self.thread_run = True
		
	def from_pic(self):
		self.thread_run = False
		self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
		if self.pic_path:
			img_bgr = predict.imreadex(self.pic_path)
			self.imgtk = self.get_imgtk(img_bgr)
			self.image_ctl.configure(image=self.imgtk)
			resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
			for resize_rate in resize_rates:
				print("resize_rate:", resize_rate)
				r, roi, color = self.predictor.predict(img_bgr, resize_rate)
				if r:
					break
			#r, roi, color = self.predictor.predict(img_bgr, 1)
			self.show_roi(r, roi, color)

	@staticmethod
	def vedio_thread(self):
		self.thread_run = True
		predict_time = time.time()
		while self.thread_run:
			_, img_bgr = self.camera.read()
			self.imgtk = self.get_imgtk(img_bgr)
			self.image_ctl.configure(image=self.imgtk)
			if time.time() - predict_time > 2:
				r, roi, color = self.predictor.predict(img_bgr)
				self.show_roi(r, roi, color)
				predict_time = time.time()
		print("run end")
		
		
def close_window():
	print("destroy")
	if surface.thread_run :
		surface.thread_run = False
		surface.thread.join(2.0)
	win.destroy()
	
	
if __name__ == '__main__':
	win=tk.Tk()
	
	surface = Surface(win)
	win.protocol('WM_DELETE_WINDOW', close_window)
	win.mainloop()
	



# In[4]:


import cv2
import dlib
import time
import threading
import math

# 提取车辆的Harr特征
# 加载车辆识别的分类器
carCascade = cv2.CascadeClassifier('myhaar.xml')

# 读取视频文件
video = cv2.VideoCapture('cars2.mp4')

# 定义初始初始值
WIDTH = 720
HEIGHT = 560
carWidht = 1.85

# 定义速度测算函数
def estimateSpeed(location1, location2, mySpeed,fps):
	# 计算像素距离
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	ppm = location2[2] / carWidht
	d_meters = d_pixels / ppm
	speed = mySpeed + d_meters * fps
	speed = speed/3
	return speed
	
def trackMultipleObjects():
	rectangleColor = (0, 0, 255)
	frameCounter = 0
	currentCarID = 0
	fps = 0
	
	carTracker = {}
	carNumbers = {}
	carLocation1 = {}
	carLocation2 = {}

	# 写入文件
	out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))
	
	while True:
		# 读取视频帧
		start_time = time.time()
		rc, image = video.read()
		# 检查是否到达视频末尾
		if type(image) == type(None):
			break

		# 转换帧的大小，以加快处理速度
		image = cv2.resize(image, (WIDTH, HEIGHT))
		resultImage = image.copy()
		
		frameCounter = frameCounter + 1
		
		carIDtoDelete = []

		# 建立追踪目标
		for carID in carTracker.keys():
			trackingQuality = carTracker[carID].update(image)
			
			if trackingQuality < 7:
				carIDtoDelete.append(carID)
				
		for carID in carIDtoDelete:
			print('Removing carID ' + str(carID) + ' from list of trackers.')
			print('Removing carID ' + str(carID) + ' previous location.')
			print('Removing carID ' + str(carID) + ' current location.')
			carTracker.pop(carID, None)
			carLocation1.pop(carID, None)
			carLocation2.pop(carID, None)

		if not (frameCounter % 10):
			# 将图像转换成灰度图像
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# 检测视频中的车辆，并用vector保存车辆的坐标、大小（用矩形表示）
			# x,y表示第n帧第i个运动目标外接矩形的中心横坐标和纵坐标位置，该坐标可以大致描述车辆目标所在的位置。
			# w,h表示第n帧第i个运动目标外接矩形的宽度和长度，可以描述车辆目标的大小
			cars = carCascade.detectMultiScale(gray, 1.1, 13, 0, (24, 24))

			# 车辆检测
			for (_x, _y, _w, _h) in cars:
				x = int(_x)
				y = int(_y)
				w = int(_w)
				h = int(_h)
			
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				matchCarID = None
			
				for carID in carTracker.keys():
					trackedPosition = carTracker[carID].get_position()
					
					t_x = int(trackedPosition.left())
					t_y = int(trackedPosition.top())
					t_w = int(trackedPosition.width())
					t_h = int(trackedPosition.height())
					
					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h
				
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = carID
				
				if matchCarID is None:
					print('Creating new tracker ' + str(currentCarID))
					# 构造追踪器
					tracker = dlib.correlation_tracker()
					# 设置追踪器的初始位置
					# 如果识别出车辆，会以Rect(x,y,w,h)的形式返回车辆的位置，然后我们可以用一个矩形网格标识车辆
					tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

					carTracker[currentCarID] = tracker
					carLocation1[currentCarID] = [x, y, w, h]

					currentCarID = currentCarID + 1
					
		for carID in carTracker.keys():
			# 获得追踪器的当前位置
			trackedPosition = carTracker[carID].get_position()
					
			t_x = int(trackedPosition.left())
			t_y = int(trackedPosition.top())
			t_w = int(trackedPosition.width())
			t_h = int(trackedPosition.height())
			
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 2)
			
			# speed estimation
			carLocation2[carID] = [t_x, t_y, t_w, t_h]

		# 计算时间差
		end_time = time.time()

		# 计算帧率
		if not (end_time == start_time):
			fps = 1.0/(end_time - start_time)
		cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

		# 计算车速
		for i in carLocation1.keys():
			if frameCounter % 10 == 0:
				[x1, y1, w1, h1] = carLocation1[i]
				[x2, y2, w2, h2] = carLocation2[i]
				# print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
				carLocation1[i] = [x2, y2, w2, h2]
				# print 'new previous location: ' + str(carLocation1[i])
				if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
					speed = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2], 100, fps)
					print('CarID ' + str(i) + ' speed is ' + str("%.2f" % round(speed, 2)) + ' km/h.\n')
					cv2.putText(resultImage, str(int(speed)) + "km/hr", (int(t_x + t_w / 2), int(t_y)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		cv2.imshow('result', resultImage)
				
		if cv2.waitKey(33) == 27:
			break
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	trackMultipleObjects()

