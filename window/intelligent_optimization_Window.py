# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'IO_Window.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1432, 791)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.button_GA = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.button_GA.setFont(font)
        self.button_GA.setObjectName("button_GA")
        self.gridLayout_2.addWidget(self.button_GA, 2, 0, 1, 1)
        self.button_PSO = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.button_PSO.setFont(font)
        self.button_PSO.setObjectName("button_PSO")
        self.gridLayout_2.addWidget(self.button_PSO, 2, 2, 1, 1)
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_GA = QtWidgets.QWidget()
        self.page_GA.setObjectName("page_GA")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.page_GA)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.label_number_of_variables_GA = QtWidgets.QLabel(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_number_of_variables_GA.setFont(font)
        self.label_number_of_variables_GA.setObjectName("label_number_of_variables_GA")
        self.horizontalLayout_15.addWidget(self.label_number_of_variables_GA)
        self.number_of_variables_GA = QtWidgets.QLineEdit(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.number_of_variables_GA.setFont(font)
        self.number_of_variables_GA.setText("")
        self.number_of_variables_GA.setObjectName("number_of_variables_GA")
        self.horizontalLayout_15.addWidget(self.number_of_variables_GA)
        self.label_NIND_GA = QtWidgets.QLabel(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_NIND_GA.setFont(font)
        self.label_NIND_GA.setObjectName("label_NIND_GA")
        self.horizontalLayout_15.addWidget(self.label_NIND_GA)
        self.NIND_GA = QtWidgets.QLineEdit(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.NIND_GA.setFont(font)
        self.NIND_GA.setObjectName("NIND_GA")
        self.horizontalLayout_15.addWidget(self.NIND_GA)
        self.label_MAXGEN_GA = QtWidgets.QLabel(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_MAXGEN_GA.setFont(font)
        self.label_MAXGEN_GA.setObjectName("label_MAXGEN_GA")
        self.horizontalLayout_15.addWidget(self.label_MAXGEN_GA)
        self.MAXGEN_GA = QtWidgets.QLineEdit(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.MAXGEN_GA.setFont(font)
        self.MAXGEN_GA.setObjectName("MAXGEN_GA")
        self.horizontalLayout_15.addWidget(self.MAXGEN_GA)
        self.gridLayout_8.addLayout(self.horizontalLayout_15, 1, 0, 1, 2)
        self.label_youhuajieguotu_GA = QtWidgets.QLabel(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_youhuajieguotu_GA.setFont(font)
        self.label_youhuajieguotu_GA.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.label_youhuajieguotu_GA.setObjectName("label_youhuajieguotu_GA")
        self.gridLayout_8.addWidget(self.label_youhuajieguotu_GA, 4, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.xunlianmoxing_GA = QtWidgets.QPushButton(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.xunlianmoxing_GA.setFont(font)
        self.xunlianmoxing_GA.setObjectName("xunlianmoxing_GA")
        self.horizontalLayout_16.addWidget(self.xunlianmoxing_GA)
        self.youhuajieguo_GA = QtWidgets.QPushButton(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.youhuajieguo_GA.setFont(font)
        self.youhuajieguo_GA.setObjectName("youhuajieguo_GA")
        self.horizontalLayout_16.addWidget(self.youhuajieguo_GA)
        self.baocunmoxing_GA = QtWidgets.QPushButton(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.baocunmoxing_GA.setFont(font)
        self.baocunmoxing_GA.setObjectName("baocunmoxing_GA")
        self.horizontalLayout_16.addWidget(self.baocunmoxing_GA)
        self.daorumoxing_GA = QtWidgets.QPushButton(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.daorumoxing_GA.setFont(font)
        self.daorumoxing_GA.setObjectName("daorumoxing_GA")
        self.horizontalLayout_16.addWidget(self.daorumoxing_GA)
        self.baocunshuju_GA = QtWidgets.QPushButton(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.baocunshuju_GA.setFont(font)
        self.baocunshuju_GA.setObjectName("baocunshuju_GA")
        self.horizontalLayout_16.addWidget(self.baocunshuju_GA)
        self.gridLayout_8.addLayout(self.horizontalLayout_16, 3, 0, 1, 2)
        self.label_youhuajieguobiao_GA = QtWidgets.QLabel(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_youhuajieguobiao_GA.setFont(font)
        self.label_youhuajieguobiao_GA.setObjectName("label_youhuajieguobiao_GA")
        self.gridLayout_8.addWidget(self.label_youhuajieguobiao_GA, 4, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.youhuajieguotu_GA = QtWidgets.QGraphicsView(self.page_GA)
        self.youhuajieguotu_GA.setObjectName("youhuajieguotu_GA")
        self.gridLayout_8.addWidget(self.youhuajieguotu_GA, 5, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_GA = QtWidgets.QLabel(self.page_GA)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_GA.setFont(font)
        self.label_GA.setObjectName("label_GA")
        self.horizontalLayout.addWidget(self.label_GA)
        self.gridLayout_8.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.youhuajieguobiao_GA = QtWidgets.QTableWidget(self.page_GA)
        self.youhuajieguobiao_GA.setObjectName("youhuajieguobiao_GA")
        self.youhuajieguobiao_GA.setColumnCount(0)
        self.youhuajieguobiao_GA.setRowCount(0)
        self.gridLayout_8.addWidget(self.youhuajieguobiao_GA, 5, 1, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_lower_bounds_GA = QtWidgets.QLabel(self.page_GA)
        self.label_lower_bounds_GA.setObjectName("label_lower_bounds_GA")
        self.horizontalLayout_2.addWidget(self.label_lower_bounds_GA)
        self.lower_bounds_GA = QtWidgets.QLineEdit(self.page_GA)
        self.lower_bounds_GA.setObjectName("lower_bounds_GA")
        self.horizontalLayout_2.addWidget(self.lower_bounds_GA)
        self.label_upper_bounds_GA = QtWidgets.QLabel(self.page_GA)
        self.label_upper_bounds_GA.setObjectName("label_upper_bounds_GA")
        self.horizontalLayout_2.addWidget(self.label_upper_bounds_GA)
        self.upper_bounds_GA = QtWidgets.QLineEdit(self.page_GA)
        self.upper_bounds_GA.setObjectName("upper_bounds_GA")
        self.horizontalLayout_2.addWidget(self.upper_bounds_GA)
        self.label_precision_GA = QtWidgets.QLabel(self.page_GA)
        self.label_precision_GA.setObjectName("label_precision_GA")
        self.horizontalLayout_2.addWidget(self.label_precision_GA)
        self.precision_GA = QtWidgets.QLineEdit(self.page_GA)
        self.precision_GA.setObjectName("precision_GA")
        self.horizontalLayout_2.addWidget(self.precision_GA)
        self.gridLayout_8.addLayout(self.horizontalLayout_2, 2, 0, 1, 2)
        self.stackedWidget.addWidget(self.page_GA)
        self.page_DE = QtWidgets.QWidget()
        self.page_DE.setObjectName("page_DE")
        self.gridLayout = QtWidgets.QGridLayout(self.page_DE)
        self.gridLayout.setObjectName("gridLayout")
        self.label_youhuajieguotu_DE = QtWidgets.QLabel(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_youhuajieguotu_DE.setFont(font)
        self.label_youhuajieguotu_DE.setAlignment(QtCore.Qt.AlignCenter)
        self.label_youhuajieguotu_DE.setObjectName("label_youhuajieguotu_DE")
        self.gridLayout.addWidget(self.label_youhuajieguotu_DE, 4, 0, 1, 1)
        self.youhuajieguotu_DE = QtWidgets.QGraphicsView(self.page_DE)
        self.youhuajieguotu_DE.setObjectName("youhuajieguotu_DE")
        self.gridLayout.addWidget(self.youhuajieguotu_DE, 5, 0, 1, 1)
        self.label_youhuajieguobiao_DE = QtWidgets.QLabel(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_youhuajieguobiao_DE.setFont(font)
        self.label_youhuajieguobiao_DE.setAlignment(QtCore.Qt.AlignCenter)
        self.label_youhuajieguobiao_DE.setObjectName("label_youhuajieguobiao_DE")
        self.gridLayout.addWidget(self.label_youhuajieguobiao_DE, 4, 1, 1, 1)
        self.youhuajieguobiao_DE = QtWidgets.QTableWidget(self.page_DE)
        self.youhuajieguobiao_DE.setObjectName("youhuajieguobiao_DE")
        self.youhuajieguobiao_DE.setColumnCount(0)
        self.youhuajieguobiao_DE.setRowCount(0)
        self.gridLayout.addWidget(self.youhuajieguobiao_DE, 5, 1, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.xunlianmoxing_DE = QtWidgets.QPushButton(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.xunlianmoxing_DE.setFont(font)
        self.xunlianmoxing_DE.setObjectName("xunlianmoxing_DE")
        self.horizontalLayout_4.addWidget(self.xunlianmoxing_DE)
        self.youhuajieguo_DE = QtWidgets.QPushButton(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.youhuajieguo_DE.setFont(font)
        self.youhuajieguo_DE.setObjectName("youhuajieguo_DE")
        self.horizontalLayout_4.addWidget(self.youhuajieguo_DE)
        self.baocunmoxing_DE = QtWidgets.QPushButton(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.baocunmoxing_DE.setFont(font)
        self.baocunmoxing_DE.setObjectName("baocunmoxing_DE")
        self.horizontalLayout_4.addWidget(self.baocunmoxing_DE)
        self.daorumoxing_DE = QtWidgets.QPushButton(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.daorumoxing_DE.setFont(font)
        self.daorumoxing_DE.setObjectName("daorumoxing_DE")
        self.horizontalLayout_4.addWidget(self.daorumoxing_DE)
        self.baocunshuju_DE = QtWidgets.QPushButton(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.baocunshuju_DE.setFont(font)
        self.baocunshuju_DE.setObjectName("baocunshuju_DE")
        self.horizontalLayout_4.addWidget(self.baocunshuju_DE)
        self.gridLayout.addLayout(self.horizontalLayout_4, 3, 0, 1, 2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_number_of_variables_DE = QtWidgets.QLabel(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_number_of_variables_DE.setFont(font)
        self.label_number_of_variables_DE.setObjectName("label_number_of_variables_DE")
        self.horizontalLayout_3.addWidget(self.label_number_of_variables_DE)
        self.number_of_variables_DE = QtWidgets.QLineEdit(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.number_of_variables_DE.setFont(font)
        self.number_of_variables_DE.setText("")
        self.number_of_variables_DE.setObjectName("number_of_variables_DE")
        self.horizontalLayout_3.addWidget(self.number_of_variables_DE)
        self.label_NIND_DE = QtWidgets.QLabel(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_NIND_DE.setFont(font)
        self.label_NIND_DE.setObjectName("label_NIND_DE")
        self.horizontalLayout_3.addWidget(self.label_NIND_DE)
        self.NIND_DE = QtWidgets.QLineEdit(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.NIND_DE.setFont(font)
        self.NIND_DE.setObjectName("NIND_DE")
        self.horizontalLayout_3.addWidget(self.NIND_DE)
        self.label_MAXGEN_DE = QtWidgets.QLabel(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label_MAXGEN_DE.setFont(font)
        self.label_MAXGEN_DE.setObjectName("label_MAXGEN_DE")
        self.horizontalLayout_3.addWidget(self.label_MAXGEN_DE)
        self.MAXGEN_DE = QtWidgets.QLineEdit(self.page_DE)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.MAXGEN_DE.setFont(font)
        self.MAXGEN_DE.setObjectName("MAXGEN_DE")
        self.horizontalLayout_3.addWidget(self.MAXGEN_DE)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_DE = QtWidgets.QLabel(self.page_DE)
        self.label_DE.setObjectName("label_DE")
        self.horizontalLayout_5.addWidget(self.label_DE)
        self.gridLayout.addLayout(self.horizontalLayout_5, 0, 0, 1, 1)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_lower_bounds_DE = QtWidgets.QLabel(self.page_DE)
        self.label_lower_bounds_DE.setObjectName("label_lower_bounds_DE")
        self.horizontalLayout_9.addWidget(self.label_lower_bounds_DE)
        self.lower_bounds_DE = QtWidgets.QLineEdit(self.page_DE)
        self.lower_bounds_DE.setObjectName("lower_bounds_DE")
        self.horizontalLayout_9.addWidget(self.lower_bounds_DE)
        self.label_upper_bounds_DE = QtWidgets.QLabel(self.page_DE)
        self.label_upper_bounds_DE.setObjectName("label_upper_bounds_DE")
        self.horizontalLayout_9.addWidget(self.label_upper_bounds_DE)
        self.upper_bounds_DE = QtWidgets.QLineEdit(self.page_DE)
        self.upper_bounds_DE.setObjectName("upper_bounds_DE")
        self.horizontalLayout_9.addWidget(self.upper_bounds_DE)
        self.gridLayout.addLayout(self.horizontalLayout_9, 2, 0, 1, 2)
        self.stackedWidget.addWidget(self.page_DE)
        self.page_PSO = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(6)
        sizePolicy.setHeightForWidth(self.page_PSO.sizePolicy().hasHeightForWidth())
        self.page_PSO.setSizePolicy(sizePolicy)
        self.page_PSO.setObjectName("page_PSO")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.page_PSO)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.youhuajieguotu_PSO = QtWidgets.QGraphicsView(self.page_PSO)
        self.youhuajieguotu_PSO.setObjectName("youhuajieguotu_PSO")
        self.gridLayout_6.addWidget(self.youhuajieguotu_PSO, 6, 0, 1, 1)
        self.label_youhuajieguobiao_PSO = QtWidgets.QLabel(self.page_PSO)
        self.label_youhuajieguobiao_PSO.setAlignment(QtCore.Qt.AlignCenter)
        self.label_youhuajieguobiao_PSO.setObjectName("label_youhuajieguobiao_PSO")
        self.gridLayout_6.addWidget(self.label_youhuajieguobiao_PSO, 5, 1, 1, 1)
        self.label_youhuajieguotu_PSO = QtWidgets.QLabel(self.page_PSO)
        self.label_youhuajieguotu_PSO.setAlignment(QtCore.Qt.AlignCenter)
        self.label_youhuajieguotu_PSO.setObjectName("label_youhuajieguotu_PSO")
        self.gridLayout_6.addWidget(self.label_youhuajieguotu_PSO, 5, 0, 1, 1)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.xunlianmoxing_PSO = QtWidgets.QPushButton(self.page_PSO)
        self.xunlianmoxing_PSO.setObjectName("xunlianmoxing_PSO")
        self.horizontalLayout_8.addWidget(self.xunlianmoxing_PSO)
        self.youhuajieguo_PSO = QtWidgets.QPushButton(self.page_PSO)
        self.youhuajieguo_PSO.setObjectName("youhuajieguo_PSO")
        self.horizontalLayout_8.addWidget(self.youhuajieguo_PSO)
        self.baocunmoxing_PSO = QtWidgets.QPushButton(self.page_PSO)
        self.baocunmoxing_PSO.setObjectName("baocunmoxing_PSO")
        self.horizontalLayout_8.addWidget(self.baocunmoxing_PSO)
        self.daorumoxing_PSO = QtWidgets.QPushButton(self.page_PSO)
        self.daorumoxing_PSO.setObjectName("daorumoxing_PSO")
        self.horizontalLayout_8.addWidget(self.daorumoxing_PSO)
        self.baocunshuju_PSO = QtWidgets.QPushButton(self.page_PSO)
        self.baocunshuju_PSO.setObjectName("baocunshuju_PSO")
        self.horizontalLayout_8.addWidget(self.baocunshuju_PSO)
        self.gridLayout_6.addLayout(self.horizontalLayout_8, 4, 0, 1, 2)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_PSO = QtWidgets.QLabel(self.page_PSO)
        self.label_PSO.setObjectName("label_PSO")
        self.horizontalLayout_6.addWidget(self.label_PSO)
        self.gridLayout_6.addLayout(self.horizontalLayout_6, 0, 0, 1, 2)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_number_of_variables_PSO = QtWidgets.QLabel(self.page_PSO)
        self.label_number_of_variables_PSO.setObjectName("label_number_of_variables_PSO")
        self.horizontalLayout_7.addWidget(self.label_number_of_variables_PSO)
        self.number_of_variables_PSO = QtWidgets.QLineEdit(self.page_PSO)
        self.number_of_variables_PSO.setText("")
        self.number_of_variables_PSO.setObjectName("number_of_variables_PSO")
        self.horizontalLayout_7.addWidget(self.number_of_variables_PSO)
        self.label_NIND_PSO = QtWidgets.QLabel(self.page_PSO)
        self.label_NIND_PSO.setObjectName("label_NIND_PSO")
        self.horizontalLayout_7.addWidget(self.label_NIND_PSO)
        self.NIND_PSO = QtWidgets.QLineEdit(self.page_PSO)
        self.NIND_PSO.setObjectName("NIND_PSO")
        self.horizontalLayout_7.addWidget(self.NIND_PSO)
        self.label_MAXGEN_PSO = QtWidgets.QLabel(self.page_PSO)
        self.label_MAXGEN_PSO.setObjectName("label_MAXGEN_PSO")
        self.horizontalLayout_7.addWidget(self.label_MAXGEN_PSO)
        self.MAXGEN_PSO = QtWidgets.QLineEdit(self.page_PSO)
        self.MAXGEN_PSO.setObjectName("MAXGEN_PSO")
        self.horizontalLayout_7.addWidget(self.MAXGEN_PSO)
        self.gridLayout_6.addLayout(self.horizontalLayout_7, 1, 0, 1, 2)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_lower_bounds_PSO = QtWidgets.QLabel(self.page_PSO)
        self.label_lower_bounds_PSO.setObjectName("label_lower_bounds_PSO")
        self.horizontalLayout_10.addWidget(self.label_lower_bounds_PSO)
        self.lower_bounds_PSO = QtWidgets.QLineEdit(self.page_PSO)
        self.lower_bounds_PSO.setObjectName("lower_bounds_PSO")
        self.horizontalLayout_10.addWidget(self.lower_bounds_PSO)
        self.label_upper_bounds_PSO = QtWidgets.QLabel(self.page_PSO)
        self.label_upper_bounds_PSO.setObjectName("label_upper_bounds_PSO")
        self.horizontalLayout_10.addWidget(self.label_upper_bounds_PSO)
        self.upper_bounds_PSO = QtWidgets.QLineEdit(self.page_PSO)
        self.upper_bounds_PSO.setObjectName("upper_bounds_PSO")
        self.horizontalLayout_10.addWidget(self.upper_bounds_PSO)
        self.gridLayout_6.addLayout(self.horizontalLayout_10, 2, 0, 1, 2)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_inertia_PSO = QtWidgets.QLabel(self.page_PSO)
        self.label_inertia_PSO.setObjectName("label_inertia_PSO")
        self.horizontalLayout_11.addWidget(self.label_inertia_PSO)
        self.inertia_PSO = QtWidgets.QLineEdit(self.page_PSO)
        self.inertia_PSO.setObjectName("inertia_PSO")
        self.horizontalLayout_11.addWidget(self.inertia_PSO)
        self.label_personal_best_parameter_PSO = QtWidgets.QLabel(self.page_PSO)
        self.label_personal_best_parameter_PSO.setObjectName("label_personal_best_parameter_PSO")
        self.horizontalLayout_11.addWidget(self.label_personal_best_parameter_PSO)
        self.personal_best_parameter_PSO = QtWidgets.QLineEdit(self.page_PSO)
        self.personal_best_parameter_PSO.setObjectName("personal_best_parameter_PSO")
        self.horizontalLayout_11.addWidget(self.personal_best_parameter_PSO)
        self.label_global_best_parameter_PSO = QtWidgets.QLabel(self.page_PSO)
        self.label_global_best_parameter_PSO.setObjectName("label_global_best_parameter_PSO")
        self.horizontalLayout_11.addWidget(self.label_global_best_parameter_PSO)
        self.global_best_parameter_PSO = QtWidgets.QLineEdit(self.page_PSO)
        self.global_best_parameter_PSO.setObjectName("global_best_parameter_PSO")
        self.horizontalLayout_11.addWidget(self.global_best_parameter_PSO)
        self.gridLayout_6.addLayout(self.horizontalLayout_11, 3, 0, 1, 2)
        self.youhuajieguobiao_PSO = QtWidgets.QTableWidget(self.page_PSO)
        self.youhuajieguobiao_PSO.setObjectName("youhuajieguobiao_PSO")
        self.youhuajieguobiao_PSO.setColumnCount(0)
        self.youhuajieguobiao_PSO.setRowCount(0)
        self.gridLayout_6.addWidget(self.youhuajieguobiao_PSO, 6, 1, 1, 1)
        self.stackedWidget.addWidget(self.page_PSO)
        self.page_RIVI = QtWidgets.QWidget()
        self.page_RIVI.setObjectName("page_RIVI")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.page_RIVI)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.stackedWidget.addWidget(self.page_RIVI)
        self.page_LASSO = QtWidgets.QWidget()
        self.page_LASSO.setObjectName("page_LASSO")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.page_LASSO)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.stackedWidget.addWidget(self.page_LASSO)
        self.gridLayout_2.addWidget(self.stackedWidget, 1, 0, 1, 4)
        self.button_DE = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.button_DE.setFont(font)
        self.button_DE.setObjectName("button_DE")
        self.gridLayout_2.addWidget(self.button_DE, 2, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1432, 30))
        self.menubar.setObjectName("menubar")
        self.menu_1 = QtWidgets.QMenu(self.menubar)
        self.menu_1.setObjectName("menu_1")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.definite_aim_func = QtWidgets.QAction(MainWindow)
        self.definite_aim_func.setObjectName("definite_aim_func")
        self.load_test = QtWidgets.QAction(MainWindow)
        self.load_test.setObjectName("load_test")
        self.testdata = QtWidgets.QAction(MainWindow)
        self.testdata.setObjectName("testdata")
        self.menu_1.addAction(self.definite_aim_func)
        self.menubar.addAction(self.menu_1.menuAction())

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_GA.setText(_translate("MainWindow", "GA"))
        self.button_PSO.setText(_translate("MainWindow", "PSO"))
        self.label_number_of_variables_GA.setToolTip(_translate("MainWindow", "目标维度"))
        self.label_number_of_variables_GA.setText(_translate("MainWindow", "控制变量个数:"))
        self.number_of_variables_GA.setPlaceholderText(_translate("MainWindow", "请输入正整数"))
        self.label_NIND_GA.setText(_translate("MainWindow", "种群规模:"))
        self.NIND_GA.setText(_translate("MainWindow", "50"))
        self.label_MAXGEN_GA.setText(_translate("MainWindow", "最大遗传代数:"))
        self.MAXGEN_GA.setText(_translate("MainWindow", "1000"))
        self.label_youhuajieguotu_GA.setText(_translate("MainWindow", "优化结果图"))
        self.xunlianmoxing_GA.setToolTip(_translate("MainWindow", "对数据进行PCA建模"))
        self.xunlianmoxing_GA.setText(_translate("MainWindow", "训练模型"))
        self.youhuajieguo_GA.setToolTip(_translate("MainWindow", "提取训练数据的PCA特征"))
        self.youhuajieguo_GA.setText(_translate("MainWindow", "优化结果"))
        self.baocunmoxing_GA.setText(_translate("MainWindow", "保存模型"))
        self.daorumoxing_GA.setText(_translate("MainWindow", "导入模型"))
        self.baocunshuju_GA.setToolTip(_translate("MainWindow", "保存特征数据"))
        self.baocunshuju_GA.setText(_translate("MainWindow", "保存数据"))
        self.label_youhuajieguobiao_GA.setText(_translate("MainWindow", "优化结果表"))
        self.label_GA.setText(_translate("MainWindow", "遗传算法"))
        self.label_lower_bounds_GA.setText(_translate("MainWindow", "控制变量下界:"))
        self.label_upper_bounds_GA.setText(_translate("MainWindow", "控制变量上界:"))
        self.label_precision_GA.setText(_translate("MainWindow", "精度:"))
        self.precision_GA.setText(_translate("MainWindow", "1e-7"))
        self.label_youhuajieguotu_DE.setText(_translate("MainWindow", "优化结果图"))
        self.label_youhuajieguobiao_DE.setText(_translate("MainWindow", "优化结果表"))
        self.xunlianmoxing_DE.setText(_translate("MainWindow", "训练模型"))
        self.youhuajieguo_DE.setText(_translate("MainWindow", "优化结果"))
        self.baocunmoxing_DE.setText(_translate("MainWindow", "保存模型"))
        self.daorumoxing_DE.setText(_translate("MainWindow", "导入模型"))
        self.baocunshuju_DE.setText(_translate("MainWindow", "保存数据"))
        self.label_number_of_variables_DE.setText(_translate("MainWindow", "控制变量个数:"))
        self.number_of_variables_DE.setPlaceholderText(_translate("MainWindow", "请输入正整数"))
        self.label_NIND_DE.setText(_translate("MainWindow", "种群规模:"))
        self.NIND_DE.setText(_translate("MainWindow", "50"))
        self.label_MAXGEN_DE.setText(_translate("MainWindow", "最大遗传代数:"))
        self.MAXGEN_DE.setText(_translate("MainWindow", "1000"))
        self.label_DE.setText(_translate("MainWindow", "差分进化算法"))
        self.label_lower_bounds_DE.setText(_translate("MainWindow", "控制变量下界:"))
        self.label_upper_bounds_DE.setText(_translate("MainWindow", "控制变量上界:"))
        self.label_youhuajieguobiao_PSO.setText(_translate("MainWindow", "优化结果表"))
        self.label_youhuajieguotu_PSO.setText(_translate("MainWindow", "优化结果图"))
        self.xunlianmoxing_PSO.setText(_translate("MainWindow", "训练模型"))
        self.youhuajieguo_PSO.setText(_translate("MainWindow", "优化结果"))
        self.baocunmoxing_PSO.setText(_translate("MainWindow", "保存模型"))
        self.daorumoxing_PSO.setText(_translate("MainWindow", "导入模型"))
        self.baocunshuju_PSO.setText(_translate("MainWindow", "保存数据"))
        self.label_PSO.setText(_translate("MainWindow", "粒子群算法"))
        self.label_number_of_variables_PSO.setText(_translate("MainWindow", "控制变量个数:"))
        self.number_of_variables_PSO.setPlaceholderText(_translate("MainWindow", "请输入正整数"))
        self.label_NIND_PSO.setText(_translate("MainWindow", "种群规模:"))
        self.NIND_PSO.setText(_translate("MainWindow", "50"))
        self.label_MAXGEN_PSO.setText(_translate("MainWindow", "最大遗传代数:"))
        self.MAXGEN_PSO.setText(_translate("MainWindow", "1000"))
        self.label_lower_bounds_PSO.setText(_translate("MainWindow", "控制变量下界:"))
        self.label_upper_bounds_PSO.setText(_translate("MainWindow", "控制变量上界:"))
        self.label_inertia_PSO.setText(_translate("MainWindow", "惯性系数:"))
        self.inertia_PSO.setText(_translate("MainWindow", "0.8"))
        self.label_personal_best_parameter_PSO.setText(_translate("MainWindow", "个体最优系数:"))
        self.personal_best_parameter_PSO.setText(_translate("MainWindow", "0.5"))
        self.label_global_best_parameter_PSO.setText(_translate("MainWindow", "全局最优系数:"))
        self.global_best_parameter_PSO.setText(_translate("MainWindow", "0.5"))
        self.button_DE.setText(_translate("MainWindow", "DE"))
        self.menu_1.setTitle(_translate("MainWindow", "数据"))
        self.definite_aim_func.setText(_translate("MainWindow", "定义目标函数"))
        self.load_test.setText(_translate("MainWindow", "Testing data"))
        self.testdata.setText(_translate("MainWindow", "Test_DATA"))
