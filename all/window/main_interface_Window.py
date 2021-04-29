# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_interface_Window.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(363, 492)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(70, 150, 211, 248))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.button_feature_selection = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.button_feature_selection.setFont(font)
        self.button_feature_selection.setObjectName("button_feature_selection")
        self.verticalLayout.addWidget(self.button_feature_selection)
        self.button_feature_extraction = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.button_feature_extraction.setFont(font)
        self.button_feature_extraction.setObjectName("button_feature_extraction")
        self.verticalLayout.addWidget(self.button_feature_extraction)
        self.button_space_projection = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.button_space_projection.setFont(font)
        self.button_space_projection.setObjectName("button_space_projection")
        self.verticalLayout.addWidget(self.button_space_projection)
        self.button_cluster = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.button_cluster.setFont(font)
        self.button_cluster.setObjectName("button_cluster")
        self.verticalLayout.addWidget(self.button_cluster)
        self.button_classification = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.button_classification.setFont(font)
        self.button_classification.setObjectName("button_classification")
        self.verticalLayout.addWidget(self.button_classification)
        self.button_intelligence_optimization = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.button_intelligence_optimization.setFont(font)
        self.button_intelligence_optimization.setObjectName("button_intelligence_optimization")
        self.verticalLayout.addWidget(self.button_intelligence_optimization)
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        self.label_title.setGeometry(QtCore.QRect(10, 20, 341, 111))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setObjectName("label_title")
        self.label_backgrund = QtWidgets.QLabel(self.centralwidget)
        self.label_backgrund.setGeometry(QtCore.QRect(0, -10, 371, 471))
        self.label_backgrund.setText("")
        self.label_backgrund.setPixmap(QtGui.QPixmap("../login_picture.jpg"))
        self.label_backgrund.setScaledContents(True)
        self.label_backgrund.setObjectName("label_backgrund")
        self.label_backgrund.raise_()
        self.verticalLayoutWidget.raise_()
        self.label_title.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 363, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_feature_selection.setText(_translate("MainWindow", "特征选择"))
        self.button_feature_extraction.setText(_translate("MainWindow", "特征提取"))
        self.button_space_projection.setText(_translate("MainWindow", "空间投影"))
        self.button_cluster.setText(_translate("MainWindow", "聚类"))
        self.button_classification.setText(_translate("MainWindow", "分类"))
        self.button_intelligence_optimization.setText(_translate("MainWindow", "智能优化"))
        self.label_title.setText(_translate("MainWindow", "<html><head/><body><p>基于大数据的长型材</p><p>人工智能算法库</p></body></html>"))