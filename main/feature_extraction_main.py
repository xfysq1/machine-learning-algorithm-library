# -*- coding: utf-8 -*-
import sys
import xlrd
import xlwt
import numpy as np
import joblib

from sklearn import preprocessing
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QGraphicsScene
from window.feature_extraction_Window import Ui_MainWindow
from feature_extraction_models.mspm_principal_component_analysis import MspmPrincipalComponentAnalysis as mspm_pca
# from models.mspm_kernel_principal_component_analysis import MspmKernelPrincipalComponentAnalysis as mspm_kpca
from feature_extraction_models.mspm_partial_least_squares import MspmPartialLeastSquares as mspm_pls
from feature_extraction_models.mspm_kernel_partial_least_squares import MspmKernelPartialLeastSquares as mspm_kpls
from feature_extraction_models.autoencoder import Autoencoder as AE
from feature_extraction_models.denoise_autoencoder import DenoiseAutoencoder as DAE
from feature_extraction_models.sparse_autoencoder import SparseAutoencoder as SAE
from feature_extraction_models.label_autoencoder import LabelAutoencoder as LAE


class FS_window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(FS_window, self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle('特征提取')
        self.setStyleSheet("#Main_Window{background-color: white}")
        self.setStyleSheet("#stackedWidget{background-color: white}")

        self.traindata.triggered.connect(self.load_traindata)
        self.testdata.triggered.connect(self.load_testdata)

        # 模型界面选择
        self.button_pca.clicked.connect(self.topage_pca)
        self.button_pls.clicked.connect(self.topage_pls)
        self.button_kpca.clicked.connect(self.topage_kpca)
        self.button_kpls.clicked.connect(self.topage_kpls)
        self.button_ae.clicked.connect(self.topage_ae)
        self.button_sae.clicked.connect(self.topage_sae)
        self.button_dae.clicked.connect(self.topage_dae)
        self.button_lae.clicked.connect(self.topage_lae)

        # PCA
        self.xunlian_pca.clicked.connect(self.construct_pca_model)
        self.xunliantezheng_pca.clicked.connect(self.extact_pca_feature)
        self.baocunmoxing_pca.clicked.connect(self.savemodel_pca)
        self.daorumoxing_pca.clicked.connect(self.loadmodel_pca)
        self.ceshitezheng_pca.clicked.connect(self.testfeature_pca)
        self.baocunshujv_pca.clicked.connect(self.savedata_pca)
        ## 画图
        self.fig_pca = Figure((12, 5))  # 15, 8
        self.canvas_pca = FigureCanvas(self.fig_pca)
        #self.canvas_pca.setParent(self.pca_gongxiantu)
        self.graphicscene_pca = QGraphicsScene()
        self.graphicscene_pca.addWidget(self.canvas_pca)
        self.toolbar_pca = NavigationToolbar(self.canvas_pca, self.pca_gongxiantu)

        # PLS
        self.xunlian_pls.clicked.connect(self.construct_pls_model)
        self.xunliantezheng_pls.clicked.connect(self.extact_pls_feature)
        self.baocunmoxing_pls.clicked.connect(self.savemodel_pls)
        self.daorumoxing_pls.clicked.connect(self.loadmodel_pls)
        self.ceshitezheng_pls.clicked.connect(self.testfeature_pls)
        self.baocunshujv_pls.clicked.connect(self.savedata_pls)
        ## 画图
        self.fig_pls = Figure((12, 5))  # 15, 8
        self.canvas_pls = FigureCanvas(self.fig_pls)
        #self.canvas_pls.setParent(self.pls_gongxiantu)
        self.graphicscene_pls = QGraphicsScene()
        self.graphicscene_pls.addWidget(self.canvas_pls)
        self.toolbar_pls = NavigationToolbar(self.canvas_pls, self.pls_gongxiantu)

        # KPCA
        self.xunlian_kpca.clicked.connect(self.construct_kpca_model)
        self.xunliantezheng_kpca.clicked.connect(self.extact_kpca_feature)
        self.baocunmoxing_kpca.clicked.connect(self.savemodel_kpca)
        self.daorumoxing_kpca.clicked.connect(self.loadmodel_kpca)
        self.ceshitezheng_kpca.clicked.connect(self.testfeature_kpca)
        self.baocunshujv_kpca.clicked.connect(self.savedata_kpca)
        ## 画图
        self.fig_kpca = Figure((6, 2.5))  # 15, 8
        self.canvas_kpca = FigureCanvas(self.fig_kpca)
        #self.canvas_kpca.setParent(self.kpca_gongxiantu)
        self.graphicscene_kpca = QGraphicsScene()
        self.graphicscene_kpca.addWidget(self.canvas_kpca)
        self.toolbar_kpca = NavigationToolbar(self.canvas_kpca, self.kpca_gongxiantu)

        # KPLS
        self.xunlian_kpls.clicked.connect(self.construct_kpls_model)
        self.xunliantezheng_kpls.clicked.connect(self.extact_kpls_feature)
        self.baocunmoxing_kpls.clicked.connect(self.savemodel_kpls)
        self.daorumoxing_kpls.clicked.connect(self.loadmodel_kpls)
        self.ceshitezheng_kpls.clicked.connect(self.testfeature_kpls)
        self.baocunshujv_kpls.clicked.connect(self.savedata_kpls)
        ## 画图
        self.fig_kpls = Figure()  # 15, 8
        self.canvas_kpls = FigureCanvas(self.fig_kpls)
        #self.canvas_kpls.setParent(self.kpls_gongxiantu)
        self.graphicscene_kpls = QGraphicsScene()
        self.graphicscene_kpls.addWidget(self.canvas_kpls)
        self.toolbar_kpls = NavigationToolbar(self.canvas_kpls, self.kpls_gongxiantu)

        # AutoEncoder
        self.chushihua_ae.clicked.connect(self.initialize_ae)
        self.daorumoxin_ae.clicked.connect(self.loadmodel_ae)
        self.xunlian_ae.clicked.connect(self.train_ae)
        self.xunliantezheng_ae.clicked.connect(self.trainfeature_ae)
        self.baocunmoxin_ae.clicked.connect(self.savemodel_ae)
        self.ceshitezheng_ae.clicked.connect(self.testfeature_ae)
        self.baocunshuju_ae.clicked.connect(self.savedata_ae)
        ## 画图用
        self.fig_ae = Figure((6, 2.5))  # 15, 8
        self.canvas_ae = FigureCanvas(self.fig_ae)
        self.canvas_ae.setParent(self.history_sae)
        self.toolbar_ae = NavigationToolbar(self.canvas_ae, self.history_sae)
        
        # SparseAutoEncoder
        self.chushihua_sae.clicked.connect(self.initialize_sae)
        self.daorumoxin_sae.clicked.connect(self.loadmodel_sae)
        self.xunlian_sae.clicked.connect(self.train_sae)
        self.xunliantezheng_sae.clicked.connect(self.trainfeature_sae)
        self.baocunmoxin_sae.clicked.connect(self.savemodel_sae)
        self.ceshitezheng_sae.clicked.connect(self.testfeature_sae)
        self.baocunshuju_sae.clicked.connect(self.savedata_sae)
        self.fig_sae = Figure((6, 2.5))  # 15, 8
        self.canvas_sae = FigureCanvas(self.fig_sae)
        self.canvas_sae.setParent(self.history_sae)
        self.toolbar_sae = NavigationToolbar(self.canvas_sae, self.history_sae)
        
        # DenoiseAutoEncoder
        self.chushihua_dae.clicked.connect(self.initialize_dae)
        self.daorumoxin_dae.clicked.connect(self.loadmodel_dae)
        self.xunlian_dae.clicked.connect(self.train_dae)
        self.xunliantezheng_dae.clicked.connect(self.trainfeature_dae)
        self.baocunmoxin_dae.clicked.connect(self.savemodel_dae)
        self.ceshitezheng_dae.clicked.connect(self.testfeature_dae)
        self.baocunshuju_dae.clicked.connect(self.savedata_dae)
        self.fig_dae = Figure((6, 2.5))  # 15, 8
        self.canvas_dae = FigureCanvas(self.fig_dae)
        self.canvas_dae.setParent(self.history_dae)
        self.toolbar_dae = NavigationToolbar(self.canvas_dae, self.history_dae)

        # LabelAutoEncoder
        self.chushihua_lae.clicked.connect(self.initialize_lae)
        self.daorumoxin_lae.clicked.connect(self.loadmodel_lae)
        self.xunlian_lae.clicked.connect(self.train_lae)
        self.xunliantezheng_lae.clicked.connect(self.trainfeature_lae)
        self.baocunmoxin_lae.clicked.connect(self.savemodel_lae)
        self.ceshitezheng_lae.clicked.connect(self.testfeature_lae)
        self.baocunshuju_lae.clicked.connect(self.savedata_lae)
        self.fig_lae = Figure((6, 2.5))  # 15, 8
        self.canvas_lae = FigureCanvas(self.fig_lae)
        self.canvas_lae.setParent(self.history_lae)
        self.toolbar_lae = NavigationToolbar(self.canvas_lae, self.history_lae)



    # 界面切换
    def topage_pca(self):
        self.stackedWidget.setCurrentWidget(self.page_PCA)
    def topage_pls(self):
        self.stackedWidget.setCurrentWidget(self.page_PLS)
    def topage_kpca(self):
        self.stackedWidget.setCurrentWidget(self.page_KPCA)
    def topage_kpls(self):
        self.stackedWidget.setCurrentWidget(self.page_KPLS)
    def topage_ae(self):
        self.stackedWidget.setCurrentWidget(self.page_AE)
    def topage_sae(self):
        self.stackedWidget.setCurrentWidget(self.page_SAE)
    def topage_dae(self):
        self.stackedWidget.setCurrentWidget(self.page_DAE)
    def topage_lae(self):
        self.stackedWidget.setCurrentWidget(self.page_LAE)

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


    # PCA
    ## 训练模型
    def construct_pca_model(self):
        try:
            n_components = int("".join(filter(str.isdigit, self.n_components_pca.text())))
            nrows0, ncols = self.train_data.shape
            if (n_components > ncols):
                QMessageBox.information(self, 'Warning', '请设定小于原始维度的特征维度', QMessageBox.Ok)
            else:
                self.PCA_model = mspm_pca(self.train_data, n_components=n_components,
                                          preprocess=self.Pre_Processing_kpca.isChecked())
                self.PCA_model.construct_pca_model()
                self.statusbar.showMessage('模型已训练')
                a = self.PCA_model.extract_pca_ratio()
                index = np.arange(n_components)
                self.fig_pca.clear()
                plt = self.fig_pca.add_subplot(111)
                # print(KPCA_model0.explained_variance_ratio_)
                plt.bar(index, a, 0.2)
                self.canvas_pca.draw()
                self.pca_gongxiantu.setScene(self.graphicscene_pca)
                self.pca_gongxiantu.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)


    # 训练特征
    def extact_pca_feature(self):
        try:
            self.features_pca = self.PCA_model.extract_pca_feature(self.train_data)
            nrows, ncols = self.features_pca.shape

            self.pca_features.setRowCount(nrows)
            self.pca_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.pca_features.setItem(i, j, QTableWidgetItem(str(self.features_pca[i, j])))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_pca(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.PCA_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_pca(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 测试特征
    def testfeature_pca(self):
        try:
            n_components = int("".join(filter(str.isdigit, self.n_components_pca.text())))
            self.PCA_model = mspm_pca(self.test_data, n_components=n_components,
                                      preprocess=self.Pre_Processing_kpca.isChecked())
            self.PCA_model.construct_pca_model()
            self.features_pca = self.PCA_model.extract_pca_feature(self.test_data)
            nrows, ncols = self.features_pca.shape

            self.pca_features.setRowCount(nrows)
            self.pca_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.pca_features.setItem(i, j, QTableWidgetItem(str(self.features_pca[i, j])))
            self.statusbar.showMessage('表格所示为测试特征')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_pca(self):
        try:
            nrows, ncols = self.features_pca.shape
            workbook_pca = xlwt.Workbook()
            sheet_pca = workbook_pca.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_pca.write(i, j, float(self.features_pca[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_pca.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)


    # PLS
    ## 构建模型
    def construct_pls_model(self):
        try:
            n_components = int("".join(filter(str.isdigit, self.n_components_pls.text())))
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_pls.text())))
            nrows, ncols = self.train_data.shape
            self.x_pls = self.train_data[:, 0:(ncols - labelcol)]
            self.y_pls = self.train_data[:, (ncols - labelcol):]
            nrows0, ncols0 = self.x_pls.shape
            if n_components > ncols0:
                QMessageBox.information(self, 'Warning', '请设定小于原始维度的特征维度', QMessageBox.Ok)
            else:
                self.PLS_model = mspm_pls(self.x_pls, self.y_pls, n_components = n_components,
                                          preprocess=self.Pre_Processing_kpca.isChecked())
                self.PLS_model.construct_pls_model()
                self.statusbar.showMessage('模型已训练')
                a = self.PLS_model.pls.coef_
                nrows1, ncols1 = a.shape
                index = np.arange(nrows1)
                if ncols1 == 1:
                    a = np.array(a.reshape((nrows1)))
                    self.fig_pls.clear()
                    plt = self.fig_pls.add_subplot(111)
                    plt.bar(index, a, 0.2)
                    self.canvas_pls.draw()
                    self.pls_gongxiantu.setScene(self.graphicscene_pls)
                    self.pls_gongxiantu.show()
                else:
                    self.fig_pls.clear()
                    for i in range(ncols1):
                        plt = self.fig_pls.add_subplot(ncols1,1,i+1)
                        plt.bar(index, a[:, i])
                    self.fig_kpls.tight_layout()
                    self.canvas_pls.draw()
                    self.pls_gongxiantu.setScene(self.graphicscene_pls)
                    self.pls_gongxiantu.show()


        except:
             QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)


    def extact_pls_feature(self):
        try:
            self.features_pls = self.PLS_model.extract_pls_feature(self.x_pls)
            nrows, ncols = self.features_pls.shape
            self.pls_features.setRowCount(nrows)
            self.pls_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.pls_features.setItem(i, j, QTableWidgetItem(str(self.features_pls[i, j])))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_pls(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.PLS_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_pls(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 测试特征
    def testfeature_pls(self):
        try:
            self.features_pls = self.PLS_model.extract_pls_feature(self.test_data)
            nrows, ncols = self.features_pls.shape

            self.pls_features.setRowCount(nrows)
            self.pls_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.pls_features.setItem(i, j, QTableWidgetItem(str(self.features_pls[i, j])))
            self.statusbar.showMessage('表格所示为测试特征')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_pls(self):
        try:
            nrows, ncols = self.features_pls.shape
            workbook_pls = xlwt.Workbook()
            sheet_pls = workbook_pls.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_pls.write(i, j, float(self.features_pls[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_pls.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)


    # KPCA
    ## 训练模型
    def construct_kpca_model(self):
        try:
            n_components = int("".join(filter(str.isdigit, self.n_components_kpca.text())))
            nrows, ncols = self.train_data.shape
            nkernel_components = int("".join(filter(str.isdigit, self.nkernel_components_kpca.text())))
            if (n_components > ncols):
                QMessageBox.information(self, 'Warning', '请设定小于原始维度的特征维度', QMessageBox.Ok)
            elif (nkernel_components < ncols):
                QMessageBox.information(self, 'Warning', '请设定大于原始维度的核特征维度', QMessageBox.Ok)
            else:
                kernel = self.kernel_function_kpca.currentText()
                self.KPCA_model = mspm_kpca(self.train_data, n_components = n_components, nkernel_components = nkernel_components,
                                        kernel = kernel, preprocess = self.Pre_Processing_kpca.isChecked())
                xKernel = self.KPCA_model.convert_to_kernel(self.train_data)
                self.KPCA_model.construct_kpca_model()
                self.statusbar.showMessage('模型已训练')
                a = self.KPCA_model.extract_kpca_ratio()
                index = np.arange(n_components)
                self.fig_kpca.clear()
                plt = self.fig_kpca.add_subplot(111)
                #print(KPCA_model0.explained_variance_ratio_)
                plt.bar(index, a, 0.2)
                self.canvas_kpca.draw()
                self.kpca_gongxiantu.setScene(self.graphicscene_kpca)
                self.kpca_gongxiantu.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    ## 训练特征
    def extact_kpca_feature(self):
        try:
            self.features_kpca = self.KPCA_model.extract_kpca_feature(self.train_data)
            nrows, ncols = self.features_kpca.shape

            self.kpca_features.setRowCount(nrows)
            self.kpca_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.kpca_features.setItem(i, j, QTableWidgetItem(str(self.features_kpca[i, j])))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_kpca(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.KPCA_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_kpca(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)


    ## 测试特征
    def testfeature_kpca(self):
        try:
            self.features_kpca = self.KPCA_model.extract_kpca_feature(self.test_data)
            nrows, ncols = self.features_kpca.shape

            self.kpca_features.setRowCount(nrows)
            self.kpca_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.kpca_features.setItem(i, j, QTableWidgetItem(str(self.features_kpca[i, j])))
            self.statusbar.showMessage('表格所示为测试特征')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)



    ## 保存数据
    def savedata_kpca(self):
        try:
            nrows, ncols = self.features_kpca.shape
            workbook_kpca = xlwt.Workbook()
            sheet_kpca = workbook_kpca.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_kpca.write(i, j, float(self.features_kpca[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_kpca.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # KPLS
    ## 训练模型
    def construct_kpls_model(self):
        try:

            n_components = int("".join(filter(str.isdigit, self.n_components_kpls.text())))
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_kpls.text())))
            nrows, ncols = self.train_data.shape
            self.x_kpls = self.train_data[:, 0:(ncols - labelcol)]
            self.y_kpls = self.train_data[:, (ncols - labelcol):]
            nrows0, ncols0 = self.x_kpls.shape
            nkernel_components = int("".join(filter(str.isdigit, self.nkernel_components_kpls.text())))
            if (n_components > ncols0):
                QMessageBox.information(self, 'Warning', '请设定小于原始维度的特征维度', QMessageBox.Ok)
            elif (nkernel_components < ncols0):
                QMessageBox.information(self, 'Warning', '请设定大于原始维度的核特征维度', QMessageBox.Ok)
            else:
                kernel = self.kernel_function_kpls.currentText()
                self.KPLS_model = mspm_kpls(self.x_kpls, self.y_kpls, n_components=n_components,
                                            nkernel_components=nkernel_components,
                                            kernel=kernel, preprocess=self.Pre_Processing_kpls.isChecked())
                xKernel = self.KPLS_model.convert_to_kernel(self.x_kpls)
                self.KPLS_model.construct_kpls_model()
                self.statusbar.showMessage('模型已训练')
                a = self.KPLS_model.kpls.coef_
                nrows1, ncols1 = a.shape
                index = np.arange(nrows1)
                if ncols1 == 1:
                    a = np.array(a.reshape((nrows1)))
                    self.fig_kpls.clear()
                    plt = self.fig_kpls.add_subplot(111)
                    plt.bar(index, a)
                    self.canvas_kpls.draw()
                    self.kpls_gongxiantu.setScene(self.graphicscene_kpls)
                    self.kpls_gongxiantu.show()
                else:
                    self.fig_kpls.clear()
                    for i in range(ncols1):
                        plt = self.fig_kpls.add_subplot(ncols1, 1, i + 1)
                        plt.bar(index, a[:, i])
                    self.fig_kpls.tight_layout()
                    self.canvas_kpls.draw()
                    self.kpls_gongxiantu.setScene(self.graphicscene_kpls)
                    self.kpls_gongxiantu.show()
                    #plt.tight_layout()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    ##训练特征
    def extact_kpls_feature(self):
        try:
            self.features_kpls = self.KPLS_model.extract_kpls_feature(self.x_kpls)
            nrows, ncols = self.features_kpls.shape
            self.kpls_features.setRowCount(nrows)
            self.kpls_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.kpls_features.setItem(i, j, QTableWidgetItem(str(self.features_kpls[i, j])))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ##保存模型
    def savemodel_kpls(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.KPLS_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ##导入模型
    def loadmodel_kpls(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ##测试特征
    def testfeature_kpls(self):
        #try:
            self.features_kpls = self.KPLS_model.extract_kpls_feature(self.test_data)
            nrows, ncols = self.features_kpls.shape
            self.kpls_features.setRowCount(nrows)
            self.kpls_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.kpls_features.setItem(i, j, QTableWidgetItem(str(self.features_kpls[i, j])))
            self.statusbar.showMessage('表格所示为测试特征')
        #except:
           # QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ##保存数据
    def savedata_kpls(self):
        try:
            nrows, ncols = self.features_kpls.shape
            workbook_kpls = xlwt.Workbook()
            sheet_kpls = workbook_kpls.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_kpls.write(i, j, float(self.features_kpls[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_kpls.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)





    # AE
    ## 构建模型
    def initialize_ae(self):
        try:
            ### 数据标准化
            self.x_ae = self.train_data
            self.StandardScaler_x_ae = preprocessing.StandardScaler().fit(self.x_ae)
            self.x_ae = self.StandardScaler_x_ae.transform(self.x_ae)
            try:
                structure = []
                for i in self.structure_ae.text().split(","):
                    structure.append(int("".join(filter(str.isdigit, i))))
                self.autoencoder = AE(self.x_ae, structure)
                self.autoencoder.construct_model()
                self.statusbar.showMessage('模型已构建')
            except:
                QMessageBox.information(self, 'Warning', '请设定隐层结构', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '数据初始化失败，请重新导入数据', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_ae(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            self.autoencoder.load_model(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 重新训练
    def train_ae(self):
        try:
            self.statusbar.showMessage('模型训练中。。。')
            epoch = int("".join(filter(str.isdigit, self.epoch_ae.text())))
            batchsize = int("".join(filter(str.isdigit, self.batchsize_ae.text())))
            self.autoencoder.train_model(epochs=epoch, batch_size=batchsize,
                                         use_Earlystopping=self.earlystop_ae.isChecked())
            self.statusbar.showMessage('模型训练完成')
            self.fig_ae.clear()
            plt = self.fig_ae.add_subplot(111)
            plt.plot(self.autoencoder.history.history['loss'])
            self.canvas_ae.draw()
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 训练特征
    def trainfeature_ae(self):
        try:
            self.features_ae = self.autoencoder.get_features(self.x_ae)
            nrows, ncols = self.features_ae.shape

            self.AE_features.setRowCount(nrows)
            self.AE_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.AE_features.setItem(i, j, QTableWidgetItem(str(self.features_ae[i, j])))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_ae(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            self.autoencoder.save_model(datafile)
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 测试特征
    def testfeature_ae(self):
        try:
            self.features_ae = self.autoencoder.get_features(self.StandardScaler_x_ae.transform(self.test_data))
            nrows, ncols = self.features_ae.shape

            self.AE_features.setRowCount(nrows)
            self.AE_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.AE_features.setItem(i, j, QTableWidgetItem(str(self.features_ae[i, j])))
            self.statusbar.showMessage('表格所示为测试特征')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_ae(self):
        try:
            nrows, ncols = self.features_ae.shape
            workbook_ae = xlwt.Workbook()
            sheet_ae = workbook_ae.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_ae.write(i, j, float(self.features_ae[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_ae.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)
    
    # SAE
    ## 构建模型
    def initialize_sae(self):
        try:
            ### 数据标准化
            self.x_sae = self.train_data
            self.StandardScaler_x_sae = preprocessing.StandardScaler().fit(self.x_sae)
            self.x_sae = self.StandardScaler_x_sae.transform(self.x_sae)
            try:
                structure = []
                for i in self.structure_sae.text().split(","):
                    structure.append(int("".join(filter(str.isdigit, i))))
                self.sparseautoencoder = SAE(self.x_sae, structure)
                self.sparseautoencoder.construct_model()
                self.statusbar.showMessage('模型已构建')
            except:
                QMessageBox.information(self, 'Warning', '请设定隐层结构', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '数据初始化失败，请重新导入数据', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_sae(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            print(datafile)
            self.sparseautoencoder.load_model(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 重新训练
    def train_sae(self):
        try:
            self.statusbar.showMessage('模型训练中。。。')
            epoch = int("".join(filter(str.isdigit, self.epoch_sae.text())))
            batchsize = int("".join(filter(str.isdigit, self.batchsize_sae.text())))
            self.sparseautoencoder.train_model(epochs=epoch, batch_size=batchsize,
                                               use_Earlystopping=self.earlystop_sae.isChecked())
            self.statusbar.showMessage('模型训练完成')
            self.fig_sae.clear()
            plt = self.fig_sae.add_subplot(111)
            plt.plot(self.sparseautoencoder.history.history['loss'])
            self.canvas_sae.draw()
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 训练特征
    def trainfeature_sae(self):
        try:
            self.features_sae = self.sparseautoencoder.get_features(self.x_sae)
            nrows, ncols = self.features_sae.shape

            self.SAE_features.setRowCount(nrows)
            self.SAE_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.SAE_features.setItem(i, j, QTableWidgetItem(str(self.features_sae[i, j])))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_sae(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            self.sparseautoencoder.save_model(datafile)
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 测试特征
    def testfeature_sae(self):
        try:
            self.features_sae = self.sparseautoencoder.get_features(self.StandardScaler_x_sae.transform(self.test_data))
            nrows, ncols = self.features_sae.shape

            self.SAE_features.setRowCount(nrows)
            self.SAE_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.SAE_features.setItem(i, j, QTableWidgetItem(str(self.features_sae[i, j])))
            self.statusbar.showMessage('表格所示为测试特征')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_sae(self):
        try:
            nrows, ncols = self.features_sae.shape
            workbook_sae = xlwt.Workbook()
            sheet_sae = workbook_sae.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_sae.write(i, j, float(self.features_sae[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_sae.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)
    
    # DAE
    ## 构建模型
    def initialize_dae(self):
        try:
            ### 数据标准化
            self.x_dae = self.train_data
            self.StandardScaler_x_dae = preprocessing.StandardScaler().fit(self.x_dae)
            self.x_dae = self.StandardScaler_x_dae.transform(self.x_dae)
            try:
                structure = []
                for i in self.structure_dae.text().split(","):
                    structure.append(int("".join(filter(str.isdigit, i))))
                corrput = int("".join(filter(str.isdigit, self.corrupt_dae.text())))
                self.denoiseautoencoder = DAE(x=self.x_dae, hidden_dims=structure, corrupt_rate=corrput)
                self.denoiseautoencoder.construct_model()
                self.statusbar.showMessage('模型已构建')
            except:
                QMessageBox.information(self, 'Warning', '请设定隐层结构', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '数据初始化失败，请重新导入数据', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_dae(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            self.denoiseautoencoder.load_model(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 重新训练
    def train_dae(self):
        try:
            self.statusbar.showMessage('模型训练中。。。')
            epoch = int("".join(filter(str.isdigit, self.epoch_dae.text())))
            batchsize = int("".join(filter(str.isdigit, self.batchsize_dae.text())))
            self.denoiseautoencoder.train_model(epochs=epoch, batch_size=batchsize,
                                                use_Earlystopping=self.earlystop_sae.isChecked())
            self.statusbar.showMessage('模型训练完成')
            self.fig_dae.clear()
            plt = self.fig_dae.add_subplot(111)
            plt.plot(self.denoiseautoencoder.history.history['loss'])
            self.canvas_dae.draw()
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 训练特征
    def trainfeature_dae(self):
        try:
            self.features_dae = self.denoiseautoencoder.get_features(self.x_dae)
            nrows, ncols = self.features_dae.shape

            self.DAE_features.setRowCount(nrows)
            self.DAE_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.DAE_features.setItem(i, j, QTableWidgetItem(str(self.features_dae[i, j])))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_dae(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            self.denoiseautoencoder.save_model(datafile)
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 测试特征
    def testfeature_dae(self):
        try:
            self.features_dae = self.denoiseautoencoder.get_features(
                self.StandardScaler_x_dae.transform(self.test_data))
            nrows, ncols = self.features_dae.shape

            self.DAE_features.setRowCount(nrows)
            self.DAE_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.DAE_features.setItem(i, j, QTableWidgetItem(str(self.features_dae[i, j])))
            self.statusbar.showMessage('表格所示为测试特征')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_dae(self):
        try:
            nrows, ncols = self.features_dae.shape
            workbook_dae = xlwt.Workbook()
            sheet_dae = workbook_dae.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_dae.write(i, j, float(self.features_dae[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_dae.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # LAE
    ## 构建模型
    def initialize_lae(self):
        try:
            ### 数据标准化
            nrows, ncols = self.train_data.shape
            labelcol = int("".join(filter(str.isdigit, self.biaoqianlieshu_lae.text())))
            self.x_lae = self.train_data[:, 0:(ncols - labelcol)]
            self.y_lae = self.train_data[:, (ncols - labelcol):]

            self.StandardScaler_x_lae = preprocessing.StandardScaler().fit(self.x_lae)
            self.x_lae = self.StandardScaler_x_lae.transform(self.x_lae)

            if self.zuoyong_lae.currentText() == "分类":
                use_onehot = True
            else:
                use_onehot = False
                self.StandardScaler_y_lae = preprocessing.StandardScaler().fit(self.y_lae)
                self.y_lae = self.StandardScaler_y_lae.transform(self.y_lae)
            try:
                structure = []
                for i in self.structure_lae.text().split(","):
                    structure.append(int("".join(filter(str.isdigit, i))))
                self.labelautoencoder = LAE(x=self.x_lae, labels=self.y_lae,
                                            hidden_dims=structure, use_onehot=use_onehot)
                self.labelautoencoder.construct_model()
                self.statusbar.showMessage('模型已构建')
            except:
                QMessageBox.information(self, 'Warning', '请设定隐层结构', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '数据初始化失败，请重新导入数据', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_lae(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            self.labelautoencoder.load_model(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 重新训练
    def train_lae(self):
        try:
            self.statusbar.showMessage('模型训练中。。。')
            epoch = int("".join(filter(str.isdigit, self.epoch_lae.text())))
            batchsize = int("".join(filter(str.isdigit, self.batchsize_lae.text())))
            self.labelautoencoder.train_model(epochs=epoch, batch_size=batchsize,
                                              use_Earlystopping=self.earlystop_lae.isChecked())
            self.statusbar.showMessage('模型训练完成')
            self.fig_lae.clear()
            plt = self.fig_lae.add_subplot(111)
            plt.plot(self.labelautoencoder.history_encoder.history['loss'])
            self.canvas_lae.draw()
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 训练特征
    def trainfeature_lae(self):
        try:
            self.features_lae = self.labelautoencoder.get_features(self.x_lae)
            nrows, ncols = self.features_lae.shape

            self.LAE_features.setRowCount(nrows)
            self.LAE_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.LAE_features.setItem(i, j, QTableWidgetItem(str(self.features_lae[i, j])))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_lae(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            self.labelautoencoder.save_model(datafile)
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 测试特征
    def testfeature_lae(self):
        try:
            self.features_lae = self.labelautoencoder.get_features(self.StandardScaler_x_lae.transform(self.test_data))
            nrows, ncols = self.features_lae.shape

            self.LAE_features.setRowCount(nrows)
            self.LAE_features.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.LAE_features.setItem(i, j, QTableWidgetItem(str(self.features_lae[i, j])))
            self.statusbar.showMessage('表格所示为测试特征')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_lae(self):
        try:
            nrows, ncols = self.features_lae.shape
            workbook_lae = xlwt.Workbook()
            sheet_lae = workbook_lae.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_lae.write(i, j, float(self.features_lae[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_lae.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FS_window()
    win.show()
    sys.exit(app.exec_())
