import sys
import xlrd
import xlwt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib
from sklearn import manifold, datasets
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from matplotlib.patches import Patch
from space_projection_models.minisom import MiniSom
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QGraphicsScene
from window.space_projection_window import Ui_MainWindow
from space_projection_models.SOMM import  SOM
from space_projection_models.autoencoder import Autoencoder as AE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
class SP_window(QMainWindow,Ui_MainWindow):

    def __init__(self,parent=None):
        super(SP_window,self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('空间投影')
        self.setStyleSheet("#Main_Window{background-color: white}")
        self.setStyleSheet("#stackedWidget{background-color: white}")

        self.traindata.triggered.connect(self.load_traindata)
        self.testdata.triggered.connect(self.load_testdata)

        # 模型界面选择
        self.button_lda.clicked.connect(self.topage_lda)
        self.button_som.clicked.connect(self.topage_som)
        self.button_tsne.clicked.connect(self.topage_tsne)

        #LDA
        self.xunlianmoxing2_lda.clicked.connect(self.construct2_lda_model)
        self.xunlianmoxing3_lda.clicked.connect(self.construct3_lda_model)
        self.baocunmoxing_lda.clicked.connect(self.savemodel_lda)
        self.daorumoxing_lda.clicked.connect(self.loadmodel_lda)
        self.ceshi2_lda.clicked.connect(self.test2_lda)
        self.ceshi3_lda.clicked.connect(self.test3_lda)

        ##画图
        self.fig_lda = Figure((12, 5))  # 15, 8
        self.canvas_lda = FigureCanvas(self.fig_lda)
        self.graphicscene_lda = QGraphicsScene()
        self.graphicscene_lda.addWidget(self.canvas_lda)
        self.toolbar_lda = NavigationToolbar(self.canvas_lda, self.lda_tu)

        # #AE
        self.chushihua_ae.clicked.connect(self.initialize_ae)
        self.xunlianmoxing2_ae.clicked.connect(self.construct2_ae_model)
        self.xunlianmoxing3_ae.clicked.connect(self.construct3_ae_model)
        self.baocunmoxing_ae.clicked.connect(self.savemodel_ae)
        self.daorumoxing_ae.clicked.connect(self.loadmodel_ae)


        # #
        # 画图
        self.fig_ae = Figure((12, 5))  # 15, 8
        self.canvas_ae = FigureCanvas(self.fig_ae)
        self.graphicscene_ae = QGraphicsScene()
        self.graphicscene_ae.addWidget(self.canvas_ae)
        self.toolbar_ae = NavigationToolbar(self.canvas_ae, self.ae_tu)

        # #SOM
        self.xunlianmoxing2_som.clicked.connect(self.construct2_som_model)
        self.xunlianmoxing3_som.clicked.connect(self.construct3_som_model)
        self.baocunmoxing_som.clicked.connect(self.savemodel_som)
        self.daorumoxing_som.clicked.connect(self.loadmodel_som)
        self.ceshi2_som.clicked.connect(self.test2_som)
        self.ceshi3_som.clicked.connect(self.test3_som)

        #
        # 画图
        self.fig_som = Figure((12, 5))  # 15, 8
        self.canvas_som = FigureCanvas(self.fig_som)
        self.graphicscene_som = QGraphicsScene()
        self.graphicscene_som.addWidget(self.canvas_som)
        self.toolbar_som = NavigationToolbar(self.canvas_som, self.som_tu)

        # #t-SNE
        self.xunlianmoxing2_tsne.clicked.connect(self.construct2_tsne_model)
        self.xunlianmoxing3_tsne.clicked.connect(self.construct3_tsne_model)
        self.baocunmoxing_tsne.clicked.connect(self.savemodel_tsne)
        self.daorumoxing_tsne.clicked.connect(self.loadmodel_tsne)
        self.ceshi2_tsne.clicked.connect(self.test2_tsne)
        self.ceshi3_tsne.clicked.connect(self.test3_tsne)

        #
        # # 画图
        self.fig_tsne = Figure((12, 5))  # 15, 8
        self.canvas_tsne = FigureCanvas(self.fig_tsne)
        self.graphicscene_tsne = QGraphicsScene()
        self.graphicscene_tsne.addWidget(self.canvas_tsne)
        self.toolbar_tsne = NavigationToolbar(self.canvas_tsne, self.tsne_tu)


    #界面切换
    def topage_lda(self):
        self.stackedWidget.setCurrentWidget(self.page_lda)


    def topage_som(self):
        self.stackedWidget.setCurrentWidget(self.page_som)


    def topage_ae(self):
        self.stackedWidget.setCurrentWidget(self.page_ae)

    def topage_tsne(self):
        self.stackedWidget.setCurrentWidget(self.page_tsne)

    # 导入训练数据
    def load_traindata(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "选择训练数据")
            table = xlrd.open_workbook(datafile).sheets()[0]
            nrows = table.nrows
            ncols = table.ncols
            self.trainWidget.setRowCount(nrows)
            self.trainWidget.setColumnCount(ncols)
            self.train_data = np.zeros((nrows, ncols))

            for i in range(nrows):
                for j in range(ncols):
                    self.trainWidget.setItem(i, j, QTableWidgetItem(str(table.cell_value(i, j))))
                    self.train_data[i, j] = table.cell_value(i, j)
            self.statusbar.showMessage('训练数据已导入')
        except:
            QMessageBox.information(self, 'Warning', '数据为EXCEL表格', QMessageBox.Ok)

    # 导入测试数据
    def load_testdata(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "选择测试数据")
            table = xlrd.open_workbook(datafile).sheets()[0]
            nrows = table.nrows
            ncols = table.ncols
            self.testWidget.setRowCount(nrows)
            self.testWidget.setColumnCount(ncols)
            self.test_data = np.zeros((nrows, ncols))

            for i in range(nrows):
                for j in range(ncols):
                    self.testWidget.setItem(i, j, QTableWidgetItem(str(table.cell_value(i, j))))
                    self.test_data[i, j] = table.cell_value(i, j)
            self.statusbar.showMessage('测试数据已导入')
        except:
            QMessageBox.information(self, 'Warning', '数据为EXCEL表格', QMessageBox.Ok)


    #LDA
    ##训练模型
    def construct2_lda_model(self):
        try:

            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_lda.text())))
            nrows, ncols = self.train_data.shape
            self.x_lda = self.train_data[:, 0:(ncols - labelcol)]
            self.y_lda = self.train_data[:, (ncols - labelcol):]

            self.y_lda = self.y_lda.astype(int)
            self.y_lda = self.y_lda.flatten()

            self.target_names = ['0','1','2','3','4']
            self.LDA_model = LDA(n_components=2)
            self.X_r = self.LDA_model.fit(self.x_lda,self.y_lda).transform(self.x_lda)
            self.colors = ['navy', 'turquoise', 'darkorange','blue','azure']

            self.fig_lda.clear()
            plt = self.fig_lda.add_subplot(111)

            for color, i in zip(self.colors, [0, 1, 2, 3, 4]):
                plt.scatter(self.X_r[self.y_lda == i, 0], self.X_r[self.y_lda == i, 1], alpha=.8, color=color)
            self.fig_lda.legend(loc='best', shadow=False, scatterpoints=1)
            self.canvas_lda.draw()
            self.lda_tu.setScene(self.graphicscene_lda)
            self.lda_tu.show()

        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)
        ##训练三维
    def construct3_lda_model(self):
        try:

            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_lda.text())))
            nrows, ncols = self.train_data.shape
            self.x_lda = self.train_data[:, 0:(ncols - labelcol)]
            self.y_lda = self.train_data[:, (ncols - labelcol):]
            self.y_lda = self.y_lda.astype(int)
            self.y_lda = self.y_lda.flatten()
            self.target_names = ['0', '1', '2', '3', '4']
            self.LDA_model = LDA(n_components=3)
            self.X_r = self.LDA_model.fit(self.x_lda, self.y_lda).transform(self.x_lda)
            self.colors = ['navy', 'turquoise', 'darkorange', 'blue', 'azure']
            self.fig_lda.clear()
            ax = self.fig_lda.add_subplot(111,projection = '3d')

            for color, i in zip(self.colors,[0,1,2,3,4]):
                ax.scatter(self.X_r[self.y_lda==i, 0], self.X_r[self.y_lda==i, 1], self.X_r[self.y_lda==i, 2])
            self.fig_lda.legend(loc='best', shadow=False, scatterpoints=1)
            self.canvas_lda.draw()
            self.lda_tu.setScene(self.graphicscene_lda)
            self.lda_tu.show()

        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)
    ##保存模型
    def savemodel_lda(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.LDA_model, datafile + '.kpl')
            QMessageBox.information(self, 'message', '模型保存完毕', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)
    ## 导入模型
    def loadmodel_lda(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            QMessageBox.information(self, 'message', '模型已加载', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ##测试2
    def test2_lda(self):
        try:


            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_lda.text())))
            nrowsp, ncolsp = self.test_data.shape
            self.LDA_model = LDA(n_components=2)
            self.xp_lda = self.test_data[:, 0:(ncolsp - labelcol)]
            self.yp_lda = self.test_data[:, (ncolsp - labelcol):]
            self.yp_lda = self.yp_lda.astype(int)
            self.yp_lda = self.yp_lda.flatten()

            self.Xp_r = self.LDA_model.fit(self.xp_lda, self.yp_lda).transform(self.xp_lda)
            self.colors = ['navy', 'turquoise', 'darkorange','blue','azure']
            self.fig_lda.clear()
            plt = self.fig_lda.add_subplot(111)
            for color, i in zip(self.colors, [0, 1, 2, 3, 4]):
                plt.scatter(self.Xp_r[self.yp_lda == i, 0], self.Xp_r[self.yp_lda == i, 1], alpha=.8, color=color)
            self.fig_lda.legend(loc='best', shadow=False, scatterpoints=1)
            self.canvas_lda.draw()
            self.lda_tu.setScene(self.graphicscene_lda)
            self.lda_tu.show()

        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ##测试3
    def test3_lda(self):
        try:

            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_lda.text())))
            nrowsp, ncolsp = self.test_data.shape
            self.LDA_model = LDA(n_components=3)
            self.xp_lda = self.test_data[:, 0:(ncolsp - labelcol)]
            self.yp_lda = self.test_data[:, (ncolsp - labelcol):]
            self.yp_lda = self.yp_lda.astype(int)
            self.yp_lda = self.yp_lda.flatten()
            self.Xp_r = self.LDA_model.fit(self.xp_lda, self.yp_lda).transform(self.xp_lda)
            self.colors = ['navy', 'turquoise', 'darkorange', 'blue', 'azure']
            self.fig_lda.clear()
            ax = self.fig_lda.add_subplot(111,projection = '3d')
            for color, i in zip(self.colors,[0,1,2,3,4]):
                ax.scatter(self.Xp_r[self.yp_lda==i, 0], self.Xp_r[self.yp_lda==i, 1], self.Xp_r[self.yp_lda==i, 2])
            self.fig_lda.legend(loc='best', shadow=False, scatterpoints=1)
            self.canvas_lda.draw()
            self.lda_tu.setScene(self.graphicscene_lda)
            self.lda_tu.show()

        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)




    #SOM
    ##训练二维
    def construct2_som_model(self):
        try:
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_som.text())))
            nrows, ncols = self.train_data.shape
            self.x_som1 = self.train_data[:, 0:(ncols - labelcol)]
            self.y_som = self.train_data[:, (ncols - labelcol):]

            self.y_som = self.y_som.astype(int)
            self.y_som = self.y_som.flatten()

            self.x_som = self.x_som1
            self.StandardScaler_x_som = preprocessing.StandardScaler().fit(self.x_som)
            self.x_som = self.StandardScaler_x_som.transform(self.x_som)
            N = self.x_som.shape[0]  # 样本数量
            M = self.x_som.shape[1]  # 维度/特征数量
            size = math.ceil(np.sqrt(5 * np.sqrt(N)))
            max_iter = 200
            self.som = MiniSom(size, size, M, sigma=3, learning_rate=0.5,
                      neighborhood_function='bubble')
            self.som.pca_weights_init(self.x_som)
            self.som.train_batch(self.x_som, max_iter, verbose=False)

            self.target_names = ['0', '1', '2', '3', '4']
            self.colors = ['navy', 'turquoise', 'darkorange', 'blue', 'azure']
            self.fig_som.clear()
            plt = self.fig_som.add_subplot(111)
            for color, i in zip(self.colors, [0, 1, 2, 3, 4]):
                plt.scatter(self.x_som[self.y_som == i, 0], self.x_som[self.y_som == i, 1], alpha=.8, color=color)
            self.canvas_som.draw()
            self.som_tu.setScene(self.graphicscene_som)
            self.som_tu.show()
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ##训练三维
    def construct3_som_model(self):
        try:
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_som.text())))
            nrows, ncols = self.train_data.shape
            self.x_som1 = self.train_data[:, 0:(ncols - labelcol)]
            self.y_som = self.train_data[:, (ncols - labelcol):]

            self.y_som = self.y_som.astype(int)
            self.y_som = self.y_som.flatten()

            self.x_som = self.x_som1
            self.StandardScaler_x_som = preprocessing.StandardScaler().fit(self.x_som)
            self.x_som = self.StandardScaler_x_som.transform(self.x_som)
            N = self.x_som.shape[0]  # 样本数量
            M = self.x_som.shape[1]  # 维度/特征数量
            size = math.ceil(np.sqrt(5 * np.sqrt(N)))
            max_iter = 200
            self.som = MiniSom(size, size, M, sigma=3, learning_rate=0.5,
                      neighborhood_function='bubble')
            self.som.pca_weights_init(self.x_som)
            self.som.train_batch(self.x_som, max_iter, verbose=False)

            self.target_names = ['0', '1', '2', '3', '4']
            self.colors = ['navy', 'turquoise', 'darkorange', 'blue', 'azure']
            self.fig_som.clear()
            ax = self.fig_som.add_subplot(111,projection = '3d')
            for color, i in zip(self.colors, [0, 1, 2, 3, 4]):
                ax.scatter(self.x_som[self.y_som == i, 0], self.x_som[self.y_som == i, 1],self.x_som[self.y_som == i, 2], alpha=.8, color=color)
            self.canvas_som.draw()
            self.som_tu.setScene(self.graphicscene_som)
            self.som_tu.show()
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)


    ##保存模型
    def savemodel_som(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.som, datafile + '.kpl')
            QMessageBox.information(self, 'message', '模型保存完毕', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_som(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            QMessageBox.information(self, 'message', '模型已加载', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)


    #测试2
    def test2_som(self):
        try:
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_som.text())))
            nrows, ncols = self.test_data.shape
            self.xp_som1 = self.test_data[:, 0:(ncols - labelcol)]
            self.yp_som = self.test_data[:, (ncols - labelcol):]

            self.yp_som = self.yp_som.astype(int)
            self.yp_som = self.yp_som.flatten()

            self.xp_som = self.xp_som1
            self.StandardScaler_xp_som = preprocessing.StandardScaler().fit(self.xp_som)
            self.xp_som = self.StandardScaler_xp_som.transform(self.xp_som)
            N = self.xp_som.shape[0]  # 样本数量
            M = self.xp_som.shape[1]  # 维度/特征数量
            size = math.ceil(np.sqrt(5 * np.sqrt(N)))
            max_iter = 200
            self.som = MiniSom(size, size, M, sigma=3, learning_rate=0.5,
                          neighborhood_function='bubble')
            self.som.pca_weights_init(self.xp_som)
            self.som.train_batch(self.xp_som, max_iter, verbose=False)

            self.target_names = ['0', '1', '2', '3', '4']
            self.colors = ['navy', 'turquoise', 'darkorange', 'blue', 'azure']
            self.fig_som.clear()
            plt = self.fig_som.add_subplot(111)
            for color, i in zip(self.colors, [0, 1, 2, 3, 4]):
                plt.scatter(self.xp_som[self.yp_som == i, 0], self.xp_som[self.yp_som == i, 1], alpha=.8, color=color)
            self.canvas_som.draw()
            self.som_tu.setScene(self.graphicscene_som)
            self.som_tu.show()
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)




    #测试3
    def test3_som(self):
        try:
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_som.text())))
            nrows, ncols = self.test_data.shape
            self.xp_som1 = self.test_data[:, 0:(ncols - labelcol)]
            self.yp_som = self.test_data[:, (ncols - labelcol):]

            self.yp_som = self.yp_som.astype(int)
            self.yp_som = self.yp_som.flatten()

            self.xp_som = self.xp_som1
            self.StandardScaler_xp_som = preprocessing.StandardScaler().fit(self.xp_som)
            self.xp_som = self.StandardScaler_xp_som.transform(self.xp_som)
            N = self.xp_som.shape[0]  # 样本数量
            M = self.xp_som.shape[1]  # 维度/特征数量
            size = math.ceil(np.sqrt(5 * np.sqrt(N)))
            max_iter = 200
            self.som = MiniSom(size, size, M, sigma=3, learning_rate=0.5,
                               neighborhood_function='bubble')
            self.som.pca_weights_init(self.xp_som)
            self.som.train_batch(self.xp_som, max_iter, verbose=False)

            self.target_names = ['0', '1', '2', '3', '4']
            self.colors = ['navy', 'turquoise', 'darkorange', 'blue', 'azure']
            self.fig_som.clear()
            ax = self.fig_som.add_subplot(111,projection='3d')
            for color, i in zip(self.colors, [0, 1, 2, 3, 4]):
                ax.scatter(self.xp_som[self.yp_som == i, 0], self.xp_som[self.yp_som == i, 1],self.xp_som[self.yp_som == i, 2], alpha=.8, color=color)
            self.canvas_som.draw()
            self.som_tu.setScene(self.graphicscene_som)
            self.som_tu.show()
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    #AE
    ##初始化模型
    def initialize_ae(self):
        try:
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_ae.text())))
            nrows, ncols = self.train_data.shape
            self.x_ae1 = self.train_data[:, 0:(ncols - labelcol)]
            self.y_ae = self.train_data[:, (ncols - labelcol):]

            self.y_ae = self.y_ae.astype(int)
            self.y_ae = self.y_ae.flatten()

            self.x_ae = self.x_ae1
            self.StandardScaler_x_ae = preprocessing.StandardScaler().fit(self.x_ae)
            self.x_ae = self.StandardScaler_x_ae.transform(self.x_ae)

            try:
                structure = []
                for i in self.structure_ae.text().split(","):
                    structure.append(int("".join(filter(str.isdigit, i))))
                self.autoencoder = AE(self.x_ae, structure)
                self.autoencoder.construct_model()
                QMessageBox.information(self, 'message', '模型已构建', QMessageBox.Ok)

            except:
                QMessageBox.information(self, 'Warning', '请设定隐层结构', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '数据初始化失败，请重新导入数据', QMessageBox.Ok)

    ##训练二维模型
    def construct2_ae_model(self):
        try:

            epoch = int("".join(filter(str.isdigit, self.epoch_ae.text())))
            batchsize = int("".join(filter(str.isdigit, self.batchsize_ae.text())))
            self.autoencoder.train_model(epochs=epoch, batch_size=batchsize,
                                         use_Earlystopping=self.earlystop_ae.isChecked())
            self.target_names = ['0', '1', '2', '3', '4']
            self.colors = ['navy', 'turquoise', 'darkorange','blue','azure']
            self.fig_ae.clear()
            plt = self.fig_ae.add_subplot(111)
            for color,i in zip(self.colors,[0,1,2,3,4]):
                plt.scatter(self.x_ae[self.y_ae==i, 0], self.x_ae[self.y_ae==i, 1], alpha=.8, color=color)
            self.canvas_ae.draw()
            self.ae_tu.setScene(self.graphicscene_ae)
            self.ae_tu.show()
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ##三维展示
    def construct3_ae_model(self):
        try:
            epoch = int("".join(filter(str.isdigit, self.epoch_ae.text())))
            batchsize = int("".join(filter(str.isdigit, self.batchsize_ae.text())))
            self.autoencoder.train_model(epochs=epoch, batch_size=batchsize,
                                         use_Earlystopping=self.earlystop_ae.isChecked())
            self.target_names = ['0', '1', '2', '3', '4']
            self.colors = ['navy', 'turquoise', 'darkorange', 'blue', 'azure']
            self.fig_ae.clear()
            ax = self.fig_ae.add_subplot(111,projection = '3d')
            for color, i in zip(self.colors, [0, 1, 2, 3, 4]):
                ax.scatter(self.x_ae[self.y_ae == i, 0], self.x_ae[self.y_ae == i, 1],self.x_ae[self.y_ae==i,2],alpha=.8, color=color)

            self.canvas_ae.draw()
            self.ae_tu.setScene(self.graphicscene_ae)
            self.ae_tu.show()
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ##保存模型
    def savemodel_ae(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            self.autoencoder.save_model(datafile)
            QMessageBox.information(self, 'message', '模型保存完毕', QMessageBox.Ok)

        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_ae(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            self.autoencoder.load_model(datafile)
            QMessageBox.information(self, 'message', '模型导入完毕', QMessageBox.Ok)

        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)







    #t-SNE
    ##训练模型

    def construct2_tsne_model(self):
        try:
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_tsne.text())))
            nrows, ncols = self.train_data.shape
            self.x_tsne = self.train_data[:, 0:(ncols - labelcol)]
            self.y_tsne = self.train_data[:, (ncols - labelcol):]

            self.y_tsne = self.y_tsne.astype(int)
            self.y_tsne = self.y_tsne.flatten()

            self.target_names = ['0','1','2','3','4']
            self.colors = ['navy', 'turquoise', 'darkorange', 'blue', 'azure']
            self.TSNE_model = manifold.TSNE(n_components=2)
            self.X_tsne = self.TSNE_model.fit_transform(self.x_tsne)
            # self.x_min, self.x_max = self.X_tsne.min(0), self.X_tsne.max(0)
            # self.X_norm = (self.X_tsne - self.x_min) / (self.x_max - self.x_min)  # 归一化
            self.fig_tsne.clear()
            plt = self.fig_tsne.add_subplot(111)

            plt.scatter(self.X_tsne[:, 0], self.X_tsne[:, 1],c = self.colors)

            self.canvas_tsne.draw()
            self.tsne_tu.setScene(self.graphicscene_tsne)
            self.tsne_tu.show()

        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    ##三维
    def construct3_tsne_model(self):
        try:
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_tsne.text())))
            nrows, ncols = self.train_data.shape
            self.x_tsne = self.train_data[:, 0:(ncols - labelcol)]
            self.y_tsne = self.train_data[:, (ncols - labelcol):]

            self.y_tsne = self.y_tsne.astype(int)
            self.y_tsne = self.y_tsne.flatten()

            self.target_names = ['0','1','2','3','4']

            self.colors = ['navy', 'turquoise', 'darkorange', 'blue', 'azure']
            self.TSNE_model = manifold.TSNE(n_components=3)
            self.X_tsne = self.TSNE_model.fit_transform(self.train_data)
            # self.x_min, self.x_max = self.X_tsne.min(0), self.X_tsne.max(0)
            # self.X_norm = (self.X_tsne - self.x_min) / (self.x_max - self.x_min)  # 归一化
            self.fig_tsne.clear()
            ax = self.fig_tsne.add_subplot(111,projection = '3d')
            for color,i in zip(self.colors,[0,1,2,3,4]):
                ax.scatter(self.X_tsne[self.y_tsne==i, 0], self.X_tsne[self.y_tsne==i, 1], self.X_tsne[self.y_tsne==i, 2])


            #self.fig_tsne.legend(loc='best', shadow=False, scatterpoints=1)
            self.canvas_tsne.draw()
            self.tsne_tu.setScene(self.graphicscene_tsne)
            self.tsne_tu.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    ##保存模型
    def savemodel_tsne(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.TSNE_model, datafile + '.kpl')
            QMessageBox.information(self, 'message', '模型保存完毕', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_tsne(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            QMessageBox.information(self, 'message', '模型已加载', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    #二维测试
    def test2_tsne(self):
        try:
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_tsne.text())))
            nrows, ncols = self.test_data.shape
            self.xp_tsne = self.test_data[:, 0:(ncols - labelcol)]
            self.yp_tsne = self.test_data[:, (ncols - labelcol):]

            self.yp_tsne = self.yp_tsne.astype(int)
            self.yp_tsne = self.yp_tsne.flatten()

            self.target_names = ['0','1','2','3','4']

            self.colors = ['navy', 'turquoise', 'darkorange', 'blue', 'azure']
            self.TSNE_model = manifold.TSNE(n_components=2)

            self.Xp_r = self.TSNE_model.fit_transform(self.xp_tsne)
            self.colors = ['navy', 'turquoise', 'darkorange','blue','azure']
            self.fig_tsne.clear()
            plt = self.fig_tsne.add_subplot(111)
            plt.scatter(self.Xp_r[:, 0], self.Xp_r[:, 1],c = self.colors)
            self.fig_tsne.legend(loc='best', shadow=False, scatterpoints=1)
            self.canvas_tsne.draw()
            self.tsne_tu.setScene(self.graphicscene_tsne)
            self.tsne_tu.show()

        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    #三维测试
    def test3_tsne(self):
        try:
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_tsne.text())))
            nrows, ncols = self.test_data.shape
            self.xp_tsne = self.test_data[:, 0:(ncols - labelcol)]
            self.yp_tsne = self.test_data[:, (ncols - labelcol):]

            self.yp_tsne = self.yp_tsne.astype(int)
            self.yp_tsne = self.yp_tsne.flatten()

            self.target_names = ['0','1','2','3','4']

            self.colors = ['navy', 'turquoise', 'darkorange', 'blue', 'azure']
            self.TSNE_model = manifold.TSNE(n_components=3)

            self.Xp_r = self.TSNE_model.fit_transform(self.xp_tsne)
            self.colors = ['navy', 'turquoise', 'darkorange','blue','azure']
            self.fig_tsne.clear()
            ax = self.fig_tsne.add_subplot(111,projection = '3d')
            for color,i in zip(self.colors,[0,1,2,3,4]):
                ax.scatter(self.Xp_r[self.yp_tsne==i, 0], self.Xp_r[self.yp_tsne==i, 1], self.Xp_r[self.yp_tsne==i, 2])
            #self.fig_tsne.legend(loc='best', shadow=False, scatterpoints=1)
            self.canvas_tsne.draw()
            self.tsne_tu.setScene(self.graphicscene_tsne)
            self.tsne_tu.show()

        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

















if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SP_window()
    win.show()
    sys.exit(app.exec_())