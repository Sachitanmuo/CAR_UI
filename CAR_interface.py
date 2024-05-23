# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CAR_interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage

class Ui_CarAccidentRecognition(object):
    def setupUi(self, CarAccidentRecognition):

        
        CarAccidentRecognition.setObjectName("CarAccidentRecognition")
        self.w = 1600
        self.h = 600
        CarAccidentRecognition.resize(self.w, self.h) # The size of the window
        self.centralwidget = QtWidgets.QWidget(CarAccidentRecognition)
        self.centralwidget.setObjectName("centralwidget")

        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget) 


        # setup the upload button
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        #self.pushButton.setGeometry(QtCore.QRect(320, 460, 201, 61))
        self.pushButton.setGeometry(QtCore.QRect(10, 560, 790, 30))
        self.pushButton.setObjectName("Report Accident")
        self.gridLayout.addWidget(self.pushButton, 1, 0, 2, 1)

        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(10, 10, 390, 270))
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 0, 0)
        

        self.graphicsView_3D = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_3D.setGeometry(QtCore.QRect(800, 10, 390, 270))
        self.graphicsView_3D.setObjectName("graphicsView_3D")
        self.gridLayout.addWidget(self.graphicsView_3D, 0, 1)


        CarAccidentRecognition.setCentralWidget(self.centralwidget)
        self.retranslateUi(CarAccidentRecognition)
        QtCore.QMetaObject.connectSlotsByName(CarAccidentRecognition)
        #Added: 
        #self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        #self.graphicsView.setGeometry(QtCore.QRect(170, 100, 256, 192))
        #self.graphicsView.setObjectName("graphicsView")

        self.closeButton = QtWidgets.QPushButton(self.centralwidget)
        self.closeButton.setGeometry(QtCore.QRect(500, 560, 790, 30))
        self.closeButton.setObjectName("closeButton")
        self.closeButton.setText("Power off")
        self.closeButton.clicked.connect(CarAccidentRecognition.close)
        self.gridLayout.addWidget(self.closeButton, 1, 1, 2, 1)

        
    def retranslateUi(self, CarAccidentRecognition):
        _translate = QtCore.QCoreApplication.translate
        CarAccidentRecognition.setWindowTitle(_translate("CarAccidentRecognition", "CarAccidentRecognition"))
        self.pushButton.setText(_translate("CarAccidentRecognition", "Upload"))


    def update_bev_image(self, image):
        """Update the BEV image in the GUI."""
        image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.graphicsView.setScene(QtWidgets.QGraphicsScene())
        self.graphicsView.scene().addPixmap(pixmap)
        rect = QtCore.QRectF(pixmap.rect())
        self.graphicsView.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def update_3d_image(self, image):
        """Update the 3D bounding box image in the GUI."""
        image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.graphicsView_3D.setScene(QtWidgets.QGraphicsScene())
        self.graphicsView_3D.scene().addPixmap(pixmap)
        rect = QtCore.QRectF(pixmap.rect())
        self.graphicsView_3D.fitInView(rect, QtCore.Qt.KeepAspectRatio)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    CarAccidentRecognition = QtWidgets.QMainWindow()
    ui = Ui_CarAccidentRecognition()
    ui.setupUi(CarAccidentRecognition)
    CarAccidentRecognition.show()
    sys.exit(app.exec_())
