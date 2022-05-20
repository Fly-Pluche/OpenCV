import sys
import time
from datetime import datetime
import cv2

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from ui_main import Ui_MainWindow
from ui_reg import Ui_Form
from ui_detect import Ui_Form_2

from mul_crack_detection import detect


# 静态载入2
class mainwindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        # 大概是继承了 Ui_MainWindow 的缘故，这里直接使用 setupUI()
        self.setupUi(self)
        self.show()
        self.detect = detect
        self.label_6.setText(str(datetime.now().strftime("%Y-%m-%d")))
        self.pushButton.clicked.connect(self.login)
        self.pushButton_2.clicked.connect(self.reg_show)
        QApplication.processEvents()
        self.image1 = None
        self.image_new = None
        self._translate = QtCore.QCoreApplication.translate

    def login(self):
        fopen = open('user.txt', 'r')
        lines = fopen.readlines()
        users = []
        pwds = []
        i = 0
        for line in lines:
            i += 1
            line = line.strip('\n')  # 去掉换行符
            if i % 2 == 0:
                pwds.append(str(line))
            else:
                users.append(str(line))

        if self.lineEdit.text() == users[0] and self.lineEdit_2.text() == pwds[0]:
            self.label_4.setText("登陆成功!稍等片刻入系统")
            QApplication.processEvents()
            # time.sleep(3)
            self.close()
            self.detect_show(users[0])
        else:
            self.label_4.setText("登陆失败，请注册！")
            QApplication.processEvents()

    def open_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        self.image1 = imgName
        '''上面一行代码是弹出选择文件的对话框，第一个参数固定，第二个参数是打开后右上角显示的内容
            第三个参数是对话框显示时默认打开的目录，"." 代表程序运行目录
            第四个参数是限制可打开的文件类型。
            返回参数 imgName为G:/xxxx/xxx.jpg，imgType为*.jpg。
            此时相当于获取到了文件地址 
        '''
        imgName_cv2 = cv2.imread(imgName)
        imgName_cv2 = cv2.resize(imgName_cv2, (355, 355))

        try:
            # im0 = cv2.cvtColor(imgName_cv2, cv2.COLOR_RGB2BGR)
            im0 = imgName_cv2
            # 这里使用cv2把这张图片读取进来，也可以用QtGui.QPixmap方式。然后由于cv2读取的跟等下显示的RGB顺序不一样，所以需要转换一下顺序
            showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
            self.ui3.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
            # 然后这个时候就可以显示一张图片了。
        except Exception as e:
            print(str(e))
            self.ui3.label_3.setText("不能含有中文路径")
            QApplication.processEvents()

    def open_image_new(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        self.image_new = imgName
        print('image', imgName)
        '''上面一行代码是弹出选择文件的对话框，第一个参数固定，第二个参数是打开后右上角显示的内容
            第三个参数是对话框显示时默认打开的目录，"." 代表程序运行目录
            第四个参数是限制可打开的文件类型。
            返回参数 imgName为G:/xxxx/xxx.jpg，imgType为*.jpg。
            此时相当于获取到了文件地址 
        '''
        imgName_cv2 = cv2.imread(imgName)
        imgName_cv2 = cv2.resize(imgName_cv2, (355, 355))

        try:
            im0 = cv2.cvtColor(imgName_cv2, cv2.COLOR_RGB2BGR)
            # 这里使用cv2把这张图片读取进来，也可以用QtGui.QPixmap方式。然后由于cv2读取的跟等下显示的RGB顺序不一样，所以需要转换一下顺序
            showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
            self.ui3.label_2.setPixmap(QtGui.QPixmap.fromImage(showImage))
            # 然后这个时候就可以显示一张图片了。
        except Exception as e:
            print(str(e))
            self.ui3.label_3.setText("不能含有中文路径")
            QApplication.processEvents()

    def detect_show(self, user):
        form = QDialog()
        self.ui3 = Ui_Form_2()
        self.ui3.setupUi(form)
        # form.setWindowModality(Qt.NonModal)  # 非模态，可与其他窗口交互
        # form.setWindowModality(Qt.WindowModal)  # 窗口模态，当前未处理完，阻止与父窗口交互
        form.setWindowModality(Qt.ApplicationModal)  # 应用程序模态，阻止与任何其他窗口交互
        form.show()
        self.ui3.label_6.setText(str(datetime.now().strftime("%Y-%m-%d")))
        self.ui3.label_8.setText(str(user))
        self.ui3.pushButton_3.clicked.connect(self.fff)
        self.ui3.pushButton.clicked.connect(self.open_image)
        self.ui3.pushButton_new.clicked.connect(self.open_image_new)
        self.ui3.pushButton_2.clicked.connect(self.detect_img)

        QApplication.processEvents()
        form.exec_()

    def detect_img(self):

        if self.image1 == None or self.image_new == None:
            return
        distance_1_mean, distance_2_mean, max_distance_1, max_distance_2 = self.detect(self.image1, self.image_new)
        print('distance_1_mean,distance_2_mean,max_distance_1,max_distance_2', distance_1_mean, distance_2_mean,
              max_distance_1, max_distance_2)
        # self.ui3.label_2.setPixmap(QtGui.QPixmap.fromImage(showImage))

        self.ui3.label_image1.setText('mean distance:'+str(distance_1_mean)+'   '+'max distance:'+str(max_distance_1))
        self.ui3.label_image2.setText('mean distance:'+str(distance_2_mean)+'   '+'max distance:'+str(max_distance_2))
        self.ui3.label_change.setText('mean change:'+str(round(abs(distance_2_mean-distance_1_mean),2))+'   '+'max change:'+str(round(abs(max_distance_2-max_distance_1),2)))

        self.image1 = None
        self.image_new = None

    def camshow(self):
        # global self.camimg
        _, self.camimg = self.camcapture.read()
        camimg = cv2.cvtColor(self.camimg, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(camimg.data, camimg.shape[1], camimg.shape[0], QtGui.QImage.Format_RGB888)
        self.ui3.label_9.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def fff(self):
        super().__init__()
        # 大概是继承了 Ui_MainWindow 的缘故，这里直接使用 setupUI()
        self.setupUi(self)
        self.show()
        self.label_6.setText(str(datetime.now().strftime("%Y-%m-%d")))
        self.pushButton.clicked.connect(self.login)
        self.pushButton_2.clicked.connect(self.reg_show)
        QApplication.processEvents()

    def reg_show(self):
        form = QDialog()
        self.ui2 = Ui_Form()
        self.ui2.setupUi(form)
        # form.setWindowModality(Qt.NonModal)  # 非模态，可与其他窗口交互
        # form.setWindowModality(Qt.WindowModal)  # 窗口模态，当前未处理完，阻止与父窗口交互
        form.setWindowModality(Qt.ApplicationModal)  # 应用程序模态，阻止与任何其他窗口交互
        form.show()
        self.ui2.label_6.setText(str(datetime.now().strftime("%Y-%m-%d")))
        self.ui2.pushButton_2.clicked.connect(self.reg)
        QApplication.processEvents()
        form.exec_()

    def reg(self):
        if self.ui2.lineEdit.text() == '' or self.ui2.lineEdit_2.text() == '' or self.ui2.lineEdit_3.text() == '':
            self.ui2.label_4.setText("用户密码不能为空！")
        elif self.ui2.lineEdit_2.text() != self.ui2.lineEdit_3.text():
            self.ui2.label_4.setText("两次输入的密码不同，请重新输入！")
        else:
            newuser = self.ui2.lineEdit.text()
            newpwd = self.ui2.lineEdit_2.text()
            self.ui2.label_4.setText("注册成功！")
            file_handle = open('user.txt', mode='w')
            file_handle.write(newuser + '\n')
            file_handle.write(newpwd + '\n')
            file_handle.close()
        QApplication.processEvents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainwindow()
    sys.exit(app.exec_())
