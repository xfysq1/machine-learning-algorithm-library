# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'win.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1062, 922)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 10, 951, 861))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSpacing(9)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_xunlianshuju = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_xunlianshuju.setFont(font)
        self.label_xunlianshuju.setObjectName("label_xunlianshuju")
        self.verticalLayout_4.addWidget(self.label_xunlianshuju)
        self.trainWidget = QtWidgets.QTableWidget(self.layoutWidget)
        self.trainWidget.setObjectName("trainWidget")
        self.trainWidget.setColumnCount(0)
        self.trainWidget.setRowCount(0)
        self.verticalLayout_4.addWidget(self.trainWidget)
        self.horizontalLayout_8.addLayout(self.verticalLayout_4)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_ceshishuju = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_ceshishuju.setFont(font)
        self.label_ceshishuju.setObjectName("label_ceshishuju")
        self.verticalLayout_5.addWidget(self.label_ceshishuju)
        self.testWidget = QtWidgets.QTableWidget(self.layoutWidget)
        self.testWidget.setObjectName("testWidget")
        self.testWidget.setColumnCount(0)
        self.testWidget.setRowCount(0)
        self.verticalLayout_5.addWidget(self.testWidget)
        self.horizontalLayout_8.addLayout(self.verticalLayout_5)
        self.horizontalLayout_8.setStretch(0, 3)
        self.horizontalLayout_8.setStretch(1, 1)
        self.horizontalLayout_8.setStretch(2, 3)
        self.verticalLayout_6.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_jiangweifangfa = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_jiangweifangfa.setFont(font)
        self.label_jiangweifangfa.setObjectName("label_jiangweifangfa")
        self.horizontalLayout_7.addWidget(self.label_jiangweifangfa)
        self.button_lda = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.button_lda.setFont(font)
        self.button_lda.setObjectName("button_lda")
        self.horizontalLayout_7.addWidget(self.button_lda)
        self.button_som = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.button_som.setFont(font)
        self.button_som.setObjectName("button_som")
        self.horizontalLayout_7.addWidget(self.button_som)
        self.button_ae = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.button_ae.setFont(font)
        self.button_ae.setObjectName("button_ae")
        self.horizontalLayout_7.addWidget(self.button_ae)
        self.button_tsne = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.button_tsne.setFont(font)
        self.button_tsne.setObjectName("button_tsne")
        self.horizontalLayout_7.addWidget(self.button_tsne)
        self.horizontalLayout_7.setStretch(0, 1)
        self.horizontalLayout_7.setStretch(1, 3)
        self.horizontalLayout_7.setStretch(2, 3)
        self.horizontalLayout_7.setStretch(3, 3)
        self.horizontalLayout_7.setStretch(4, 3)
        self.verticalLayout_6.addLayout(self.horizontalLayout_7)
        self.stackedWidget = QtWidgets.QStackedWidget(self.layoutWidget)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_lda = QtWidgets.QWidget()
        self.page_lda.setObjectName("page_lda")
        self.layoutWidget1 = QtWidgets.QWidget(self.page_lda)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 10, 911, 441))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_biaoqianlieshu_lda = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_biaoqianlieshu_lda.setFont(font)
        self.label_biaoqianlieshu_lda.setObjectName("label_biaoqianlieshu_lda")
        self.horizontalLayout.addWidget(self.label_biaoqianlieshu_lda)
        self.biaoqianlieshu_lda = QtWidgets.QLineEdit(self.layoutWidget1)
        self.biaoqianlieshu_lda.setObjectName("biaoqianlieshu_lda")
        self.horizontalLayout.addWidget(self.biaoqianlieshu_lda)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 7)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.label_jiangweijieguo__lda = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_jiangweijieguo__lda.setFont(font)
        self.label_jiangweijieguo__lda.setObjectName("label_jiangweijieguo__lda")
        self.verticalLayout.addWidget(self.label_jiangweijieguo__lda)
        self.lda_tu = QtWidgets.QGraphicsView(self.layoutWidget1)
        self.lda_tu.setObjectName("lda_tu")
        self.verticalLayout.addWidget(self.lda_tu)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.xunlianmoxing2_lda = QtWidgets.QPushButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.xunlianmoxing2_lda.setFont(font)
        self.xunlianmoxing2_lda.setObjectName("xunlianmoxing2_lda")
        self.horizontalLayout_2.addWidget(self.xunlianmoxing2_lda)
        self.xunlianmoxing3_lda = QtWidgets.QPushButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.xunlianmoxing3_lda.setFont(font)
        self.xunlianmoxing3_lda.setObjectName("xunlianmoxing3_lda")
        self.horizontalLayout_2.addWidget(self.xunlianmoxing3_lda)
        self.baocunmoxing_lda = QtWidgets.QPushButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.baocunmoxing_lda.setFont(font)
        self.baocunmoxing_lda.setObjectName("baocunmoxing_lda")
        self.horizontalLayout_2.addWidget(self.baocunmoxing_lda)
        self.daorumoxing_lda = QtWidgets.QPushButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.daorumoxing_lda.setFont(font)
        self.daorumoxing_lda.setObjectName("daorumoxing_lda")
        self.horizontalLayout_2.addWidget(self.daorumoxing_lda)
        self.ceshi2_lda = QtWidgets.QPushButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.ceshi2_lda.setFont(font)
        self.ceshi2_lda.setObjectName("ceshi2_lda")
        self.horizontalLayout_2.addWidget(self.ceshi2_lda)
        self.ceshi3_lda = QtWidgets.QPushButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.ceshi3_lda.setFont(font)
        self.ceshi3_lda.setObjectName("ceshi3_lda")
        self.horizontalLayout_2.addWidget(self.ceshi3_lda)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.stackedWidget.addWidget(self.page_lda)
        self.page_som = QtWidgets.QWidget()
        self.page_som.setObjectName("page_som")
        self.layoutWidget_2 = QtWidgets.QWidget(self.page_som)
        self.layoutWidget_2.setGeometry(QtCore.QRect(10, 10, 911, 441))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_biaoqianlieshu_som = QtWidgets.QLabel(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_biaoqianlieshu_som.setFont(font)
        self.label_biaoqianlieshu_som.setObjectName("label_biaoqianlieshu_som")
        self.horizontalLayout_9.addWidget(self.label_biaoqianlieshu_som)
        self.biaoqianlieshu_som = QtWidgets.QLineEdit(self.layoutWidget_2)
        self.biaoqianlieshu_som.setObjectName("biaoqianlieshu_som")
        self.horizontalLayout_9.addWidget(self.biaoqianlieshu_som)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem2)
        self.horizontalLayout_9.setStretch(0, 1)
        self.horizontalLayout_9.setStretch(1, 1)
        self.horizontalLayout_9.setStretch(2, 7)
        self.verticalLayout_7.addLayout(self.horizontalLayout_9)
        self.label_jiangweijieguo__som = QtWidgets.QLabel(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_jiangweijieguo__som.setFont(font)
        self.label_jiangweijieguo__som.setObjectName("label_jiangweijieguo__som")
        self.verticalLayout_7.addWidget(self.label_jiangweijieguo__som)
        self.som_tu = QtWidgets.QGraphicsView(self.layoutWidget_2)
        self.som_tu.setObjectName("som_tu")
        self.verticalLayout_7.addWidget(self.som_tu)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.xunlianmoxing2_som = QtWidgets.QPushButton(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.xunlianmoxing2_som.setFont(font)
        self.xunlianmoxing2_som.setObjectName("xunlianmoxing2_som")
        self.horizontalLayout_10.addWidget(self.xunlianmoxing2_som)
        self.xunlianmoxing3_som = QtWidgets.QPushButton(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.xunlianmoxing3_som.setFont(font)
        self.xunlianmoxing3_som.setObjectName("xunlianmoxing3_som")
        self.horizontalLayout_10.addWidget(self.xunlianmoxing3_som)
        self.baocunmoxing_som = QtWidgets.QPushButton(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.baocunmoxing_som.setFont(font)
        self.baocunmoxing_som.setObjectName("baocunmoxing_som")
        self.horizontalLayout_10.addWidget(self.baocunmoxing_som)
        self.daorumoxing_som = QtWidgets.QPushButton(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.daorumoxing_som.setFont(font)
        self.daorumoxing_som.setObjectName("daorumoxing_som")
        self.horizontalLayout_10.addWidget(self.daorumoxing_som)
        self.ceshi2_som = QtWidgets.QPushButton(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.ceshi2_som.setFont(font)
        self.ceshi2_som.setObjectName("ceshi2_som")
        self.horizontalLayout_10.addWidget(self.ceshi2_som)
        self.ceshi3_som = QtWidgets.QPushButton(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.ceshi3_som.setFont(font)
        self.ceshi3_som.setObjectName("ceshi3_som")
        self.horizontalLayout_10.addWidget(self.ceshi3_som)
        self.verticalLayout_7.addLayout(self.horizontalLayout_10)
        self.stackedWidget.addWidget(self.page_som)
        self.page_ae = QtWidgets.QWidget()
        self.page_ae.setObjectName("page_ae")
        self.layoutWidget2 = QtWidgets.QWidget(self.page_ae)
        self.layoutWidget2.setGeometry(QtCore.QRect(10, 10, 911, 441))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_biaoqianlieshu_ae = QtWidgets.QLabel(self.layoutWidget2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_biaoqianlieshu_ae.setFont(font)
        self.label_biaoqianlieshu_ae.setObjectName("label_biaoqianlieshu_ae")
        self.horizontalLayout_3.addWidget(self.label_biaoqianlieshu_ae)
        self.biaoqianlieshu_ae = QtWidgets.QLineEdit(self.layoutWidget2)
        self.biaoqianlieshu_ae.setObjectName("biaoqianlieshu_ae")
        self.horizontalLayout_3.addWidget(self.biaoqianlieshu_ae)
        self.label_yingcejiegou_ae = QtWidgets.QLabel(self.layoutWidget2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_yingcejiegou_ae.setFont(font)
        self.label_yingcejiegou_ae.setObjectName("label_yingcejiegou_ae")
        self.horizontalLayout_3.addWidget(self.label_yingcejiegou_ae)
        self.structure_ae = QtWidgets.QLineEdit(self.layoutWidget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.structure_ae.sizePolicy().hasHeightForWidth())
        self.structure_ae.setSizePolicy(sizePolicy)
        self.structure_ae.setObjectName("structure_ae")
        self.horizontalLayout_3.addWidget(self.structure_ae)
        self.label_epoch_ae = QtWidgets.QLabel(self.layoutWidget2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_epoch_ae.setFont(font)
        self.label_epoch_ae.setObjectName("label_epoch_ae")
        self.horizontalLayout_3.addWidget(self.label_epoch_ae)
        self.epoch_ae = QtWidgets.QLineEdit(self.layoutWidget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.epoch_ae.sizePolicy().hasHeightForWidth())
        self.epoch_ae.setSizePolicy(sizePolicy)
        self.epoch_ae.setObjectName("epoch_ae")
        self.horizontalLayout_3.addWidget(self.epoch_ae)
        self.label_batchsize_ae = QtWidgets.QLabel(self.layoutWidget2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_batchsize_ae.setFont(font)
        self.label_batchsize_ae.setObjectName("label_batchsize_ae")
        self.horizontalLayout_3.addWidget(self.label_batchsize_ae)
        self.batchsize_ae = QtWidgets.QLineEdit(self.layoutWidget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.batchsize_ae.sizePolicy().hasHeightForWidth())
        self.batchsize_ae.setSizePolicy(sizePolicy)
        self.batchsize_ae.setObjectName("batchsize_ae")
        self.horizontalLayout_3.addWidget(self.batchsize_ae)
        self.earlystop_ae = QtWidgets.QRadioButton(self.layoutWidget2)
        self.earlystop_ae.setChecked(True)
        self.earlystop_ae.setAutoRepeat(False)
        self.earlystop_ae.setObjectName("earlystop_ae")
        self.horizontalLayout_3.addWidget(self.earlystop_ae)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 1)
        self.horizontalLayout_3.setStretch(3, 3)
        self.horizontalLayout_3.setStretch(4, 1)
        self.horizontalLayout_3.setStretch(5, 1)
        self.horizontalLayout_3.setStretch(6, 1)
        self.horizontalLayout_3.setStretch(7, 1)
        self.horizontalLayout_3.setStretch(8, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.label_jiangweijieguo__ae = QtWidgets.QLabel(self.layoutWidget2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_jiangweijieguo__ae.setFont(font)
        self.label_jiangweijieguo__ae.setObjectName("label_jiangweijieguo__ae")
        self.verticalLayout_2.addWidget(self.label_jiangweijieguo__ae)
        self.ae_tu = QtWidgets.QGraphicsView(self.layoutWidget2)
        self.ae_tu.setObjectName("ae_tu")
        self.verticalLayout_2.addWidget(self.ae_tu)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.chushihua_ae = QtWidgets.QPushButton(self.layoutWidget2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.chushihua_ae.setFont(font)
        self.chushihua_ae.setObjectName("chushihua_ae")
        self.horizontalLayout_4.addWidget(self.chushihua_ae)
        self.xunlianmoxing2_ae = QtWidgets.QPushButton(self.layoutWidget2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.xunlianmoxing2_ae.setFont(font)
        self.xunlianmoxing2_ae.setObjectName("xunlianmoxing2_ae")
        self.horizontalLayout_4.addWidget(self.xunlianmoxing2_ae)
        self.xunlianmoxing3_ae = QtWidgets.QPushButton(self.layoutWidget2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.xunlianmoxing3_ae.setFont(font)
        self.xunlianmoxing3_ae.setObjectName("xunlianmoxing3_ae")
        self.horizontalLayout_4.addWidget(self.xunlianmoxing3_ae)
        self.baocunmoxing_ae = QtWidgets.QPushButton(self.layoutWidget2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.baocunmoxing_ae.setFont(font)
        self.baocunmoxing_ae.setObjectName("baocunmoxing_ae")
        self.horizontalLayout_4.addWidget(self.baocunmoxing_ae)
        self.daorumoxing_ae = QtWidgets.QPushButton(self.layoutWidget2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.daorumoxing_ae.setFont(font)
        self.daorumoxing_ae.setObjectName("daorumoxing_ae")
        self.horizontalLayout_4.addWidget(self.daorumoxing_ae)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.stackedWidget.addWidget(self.page_ae)
        self.page_tsne = QtWidgets.QWidget()
        self.page_tsne.setObjectName("page_tsne")
        self.layoutWidget3 = QtWidgets.QWidget(self.page_tsne)
        self.layoutWidget3.setGeometry(QtCore.QRect(10, 10, 911, 441))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_biaoqianlieshu_tsne = QtWidgets.QLabel(self.layoutWidget3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_biaoqianlieshu_tsne.setFont(font)
        self.label_biaoqianlieshu_tsne.setObjectName("label_biaoqianlieshu_tsne")
        self.horizontalLayout_5.addWidget(self.label_biaoqianlieshu_tsne)
        self.biaoqianlieshu_tsne = QtWidgets.QLineEdit(self.layoutWidget3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.biaoqianlieshu_tsne.sizePolicy().hasHeightForWidth())
        self.biaoqianlieshu_tsne.setSizePolicy(sizePolicy)
        self.biaoqianlieshu_tsne.setObjectName("biaoqianlieshu_tsne")
        self.horizontalLayout_5.addWidget(self.biaoqianlieshu_tsne)
        spacerItem3 = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)
        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 1)
        self.horizontalLayout_5.setStretch(2, 7)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.label_jiangweijieguo_tsne = QtWidgets.QLabel(self.layoutWidget3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_jiangweijieguo_tsne.setFont(font)
        self.label_jiangweijieguo_tsne.setObjectName("label_jiangweijieguo_tsne")
        self.verticalLayout_3.addWidget(self.label_jiangweijieguo_tsne)
        self.tsne_tu = QtWidgets.QGraphicsView(self.layoutWidget3)
        self.tsne_tu.setObjectName("tsne_tu")
        self.verticalLayout_3.addWidget(self.tsne_tu)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.xunlianmoxing2_tsne = QtWidgets.QPushButton(self.layoutWidget3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.xunlianmoxing2_tsne.setFont(font)
        self.xunlianmoxing2_tsne.setObjectName("xunlianmoxing2_tsne")
        self.horizontalLayout_6.addWidget(self.xunlianmoxing2_tsne)
        self.xunlianmoxing3_tsne = QtWidgets.QPushButton(self.layoutWidget3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.xunlianmoxing3_tsne.setFont(font)
        self.xunlianmoxing3_tsne.setObjectName("xunlianmoxing3_tsne")
        self.horizontalLayout_6.addWidget(self.xunlianmoxing3_tsne)
        self.baocunmoxing_tsne = QtWidgets.QPushButton(self.layoutWidget3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.baocunmoxing_tsne.setFont(font)
        self.baocunmoxing_tsne.setObjectName("baocunmoxing_tsne")
        self.horizontalLayout_6.addWidget(self.baocunmoxing_tsne)
        self.daorumoxing_tsne = QtWidgets.QPushButton(self.layoutWidget3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.daorumoxing_tsne.setFont(font)
        self.daorumoxing_tsne.setObjectName("daorumoxing_tsne")
        self.horizontalLayout_6.addWidget(self.daorumoxing_tsne)
        self.ceshi2_tsne = QtWidgets.QPushButton(self.layoutWidget3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.ceshi2_tsne.setFont(font)
        self.ceshi2_tsne.setObjectName("ceshi2_tsne")
        self.horizontalLayout_6.addWidget(self.ceshi2_tsne)
        self.ceshi3_tsne = QtWidgets.QPushButton(self.layoutWidget3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setUnderline(True)
        self.ceshi3_tsne.setFont(font)
        self.ceshi3_tsne.setObjectName("ceshi3_tsne")
        self.horizontalLayout_6.addWidget(self.ceshi3_tsne)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.stackedWidget.addWidget(self.page_tsne)
        self.verticalLayout_6.addWidget(self.stackedWidget)
        self.verticalLayout_6.setStretch(0, 4)
        self.verticalLayout_6.setStretch(1, 1)
        self.verticalLayout_6.setStretch(2, 6)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1062, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.traindata = QtWidgets.QAction(MainWindow)
        self.traindata.setObjectName("traindata")
        self.testdata = QtWidgets.QAction(MainWindow)
        self.testdata.setObjectName("testdata")
        self.menu.addAction(self.traindata)
        self.menu.addAction(self.testdata)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_xunlianshuju.setText(_translate("MainWindow", "训练数据"))
        self.label_ceshishuju.setText(_translate("MainWindow", "测试数据"))
        self.label_jiangweifangfa.setText(_translate("MainWindow", "降维方法"))
        self.button_lda.setText(_translate("MainWindow", "LDA"))
        self.button_som.setText(_translate("MainWindow", "SOM"))
        self.button_ae.setText(_translate("MainWindow", "AE"))
        self.button_tsne.setText(_translate("MainWindow", "t-SNE"))
        self.label_biaoqianlieshu_lda.setText(_translate("MainWindow", "标签列数"))
        self.label_jiangweijieguo__lda.setText(_translate("MainWindow", "降维结果"))
        self.xunlianmoxing2_lda.setText(_translate("MainWindow", "训练二维模型"))
        self.xunlianmoxing3_lda.setText(_translate("MainWindow", "训练三维模型"))
        self.baocunmoxing_lda.setText(_translate("MainWindow", "保存模型"))
        self.daorumoxing_lda.setText(_translate("MainWindow", "导入模型"))
        self.ceshi2_lda.setText(_translate("MainWindow", "二维测试"))
        self.ceshi3_lda.setText(_translate("MainWindow", "三维测试"))
        self.label_biaoqianlieshu_som.setText(_translate("MainWindow", "标签列数"))
        self.label_jiangweijieguo__som.setText(_translate("MainWindow", "降维结果"))
        self.xunlianmoxing2_som.setText(_translate("MainWindow", "训练二维模型"))
        self.xunlianmoxing3_som.setText(_translate("MainWindow", "训练三维模型"))
        self.baocunmoxing_som.setText(_translate("MainWindow", "保存模型"))
        self.daorumoxing_som.setText(_translate("MainWindow", "导入模型"))
        self.ceshi2_som.setText(_translate("MainWindow", "二维测试"))
        self.ceshi3_som.setText(_translate("MainWindow", "三维测试"))
        self.label_biaoqianlieshu_ae.setText(_translate("MainWindow", "标签列数"))
        self.label_yingcejiegou_ae.setText(_translate("MainWindow", "隐层结构"))
        self.label_epoch_ae.setText(_translate("MainWindow", "epoch"))
        self.label_batchsize_ae.setText(_translate("MainWindow", "batch size"))
        self.earlystop_ae.setText(_translate("MainWindow", "EarlyStopping"))
        self.label_jiangweijieguo__ae.setText(_translate("MainWindow", "降维结果"))
        self.chushihua_ae.setText(_translate("MainWindow", "初始化模型"))
        self.xunlianmoxing2_ae.setText(_translate("MainWindow", "二维可视化"))
        self.xunlianmoxing3_ae.setText(_translate("MainWindow", "三维可视化"))
        self.baocunmoxing_ae.setText(_translate("MainWindow", "保存模型"))
        self.daorumoxing_ae.setText(_translate("MainWindow", "导入模型"))
        self.label_biaoqianlieshu_tsne.setText(_translate("MainWindow", "标签列数"))
        self.label_jiangweijieguo_tsne.setText(_translate("MainWindow", "降维结果"))
        self.xunlianmoxing2_tsne.setText(_translate("MainWindow", "训练二维模型"))
        self.xunlianmoxing3_tsne.setText(_translate("MainWindow", "训练三维模型"))
        self.baocunmoxing_tsne.setText(_translate("MainWindow", "保存模型"))
        self.daorumoxing_tsne.setText(_translate("MainWindow", "导入模型"))
        self.ceshi2_tsne.setText(_translate("MainWindow", "二维测试"))
        self.ceshi3_tsne.setText(_translate("MainWindow", "三维测试"))
        self.menu.setTitle(_translate("MainWindow", "数据"))
        self.traindata.setText(_translate("MainWindow", "导入训练数据"))
        self.testdata.setText(_translate("MainWindow", "导入测试数据"))