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
	# video = cv2.VideoCapture(r"D:\workspace\Yolov5-Deepsort-main\data\1.mp4")
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
