# -*- coding: utf-8 -*-

import sys
import xlrd
import xlwt
import numpy as np
import joblib

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from scipy.cluster.hierarchy import dendrogram

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QGraphicsScene
from window.cluster_Window import Ui_MainWindow

from cluster_models.kmeans import KMEANS
from cluster_models.gaussian_mixture_models import GaussianMixtureModels
from cluster_models.hierarchical_clustering import HierarchicalClustering

class JL_window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(JL_window, self).__init__(parent)

        self.setupUi(self)

        self.setWindowTitle('聚类分析')
        self.setStyleSheet("#Main_Window{background-color: white}")
        self.setStyleSheet("#stackedWidget{background-color: white}")

        self.traindata.triggered.connect(self.load_traindata)
        self.testdata.triggered.connect(self.load_testdata)

        # 聚类方法的界面选择
        self.button_kmeans.clicked.connect(self.topage_kmeans)
        self.button_gmm.clicked.connect(self.topage_gmm)
        self.button_hcc.clicked.connect(self.topage_hcc)

        # KMEANS界面的按钮
        self.xunlian_kmeans.clicked.connect(self.train_kmeans)
        self.xunlianjieguo_kmeans.clicked.connect(self.trainresult_kmeans)
        self.baocunmoxing_kmeans.clicked.connect(self.savemodel_kmeans)
        self.daorumoxing_kmeans.clicked.connect(self.loadmodel_kmeans)
        self.ceshijieguo_kmeans.clicked.connect(self.testresult_kmeans)
        self.baocunshuju_kmeans.clicked.connect(self.savedata_kmeans)

        # GMM界面的按钮
        self.xunlian_gmm.clicked.connect(self.train_gmm)
        self.xunlianjieguo_gmm.clicked.connect(self.trainresult_gmm)
        self.baocunmoxing_gmm.clicked.connect(self.savemodel_gmm)
        self.daorumoxing_gmm.clicked.connect(self.loadmodel_gmm)
        self.ceshijieguo_gmm.clicked.connect(self.testresult_gmm)
        self.baocunshuju_gmm.clicked.connect(self.savedata_gmm)
        ## 画图
        self.fig_gmm = Figure(figsize=(4,2))
        self.canvas_gmm = FigureCanvas(self.fig_gmm)
        self.graphicscene_gmm = QGraphicsScene()
        self.graphicscene_gmm.addWidget(self.canvas_gmm)
        self.toolbar_gmm = NavigationToolbar(self.canvas_gmm, self.gmm_beiyesi)

        # 层次聚类HCC界面的按钮
        self.xunlian_hcc.clicked.connect(self.train_hcc)
        self.xunlianjieguo_hcc.clicked.connect(self.trainresult_hcc)
        self.baocunmoxing_hcc.clicked.connect(self.savemodel_hcc)
        self.daorumoxing_hcc.clicked.connect(self.loadmodel_hcc)
        self.ceshijieguo_hcc.clicked.connect(self.testresult_hcc)
        self.baocunshuju_hcc.clicked.connect(self.savedata_hcc)
        ## 画图
        self.fig_hcc = Figure(figsize=(4,2))
        self.canvas_hcc = FigureCanvas(self.fig_hcc)
        self.graphicscene_hcc = QGraphicsScene()
        self.graphicscene_hcc.addWidget(self.canvas_hcc)
        self.toolbar_hcc = NavigationToolbar(self.canvas_hcc, self.hcc_shuzhuangtu)
        self.comboBox_criterion_hcc.currentIndexChanged.connect(self.change_line_hcc)


    # 界面切换
    def topage_kmeans(self):
        self.stackedWidget.setCurrentWidget(self.page_KMeans)
    def topage_gmm(self):
        self.stackedWidget.setCurrentWidget(self.page_GMM)
    def topage_hcc(self):
        self.stackedWidget.setCurrentWidget(self.page_HCC)

    def change_line_hcc(self):
        combo = self.comboBox_criterion_hcc.currentText()
        if (combo == 'inconsistent'):  # 不相关性
            self.line_criterion_hcc.setPlaceholderText('请输入0~1的数')
        elif (combo == 'distance'):  # 距离
            self.line_criterion_hcc.setPlaceholderText('请输入大于0的数')
        elif (combo == 'maxclust'):  # 聚类簇数
            self.line_criterion_hcc.setPlaceholderText('请输入正整数')
        else:
            self.line_criterion_hcc.setPlaceholderText('请输入...')


    # 导入训练数据，为纯数据，第一行是第一个样本，每列为样本特征
    def load_traindata(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "选择训练数据")
            wb = xlrd.open_workbook(datafile)
            table = wb.sheets()[0]
            nrows = table.nrows
            ncols = table.ncols
            self.trainWidget.setRowCount(nrows)
            self.trainWidget.setColumnCount(ncols)
            self.train_data = np.zeros((nrows, ncols))

            for i in range(nrows):
                for j in range(ncols):
                    self.trainWidget.setItem(i,j,QTableWidgetItem(str(table.cell_value(i, j))))
                    self.train_data[i, j] = table.cell_value(i, j)

            self.statusbar.showMessage('训练数据已导入')
        except:
            QMessageBox.information(self, 'Warning', '数据为EXCEL表格', QMessageBox.Ok)

    # 导入测试数据
    def load_testdata(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "选择测试数据")
            wb = xlrd.open_workbook(datafile)
            table = wb.sheets()[0]
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

    # KMEANS
    ## 训练模型
    def train_kmeans(self):
        try:
            n_clusters = int("".join(filter(str.isdigit, self.n_clusters_kmeans.text())))
            self.kmeans_model = KMEANS(self.train_data, n_clusters=n_clusters,
                                       preprocess=self.Pre_Processing_kmeans.isChecked())
            self.kmeans_model.construct_kmeans_model()
            self.statusbar.showMessage('模型已训练')
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    ## 训练结果
    def trainresult_kmeans(self):
        try:
            self.labels_kmeans = self.kmeans_model.kmeans.labels_
            self.labels_kmeans = np.mat(self.labels_kmeans).T
            nrows, ncols = self.labels_kmeans.shape
            self.kmeans_xunlianjieguo.setRowCount(nrows)
            self.kmeans_xunlianjieguo.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.kmeans_xunlianjieguo.setItem(i, j, QTableWidgetItem(str(self.labels_kmeans[i, j])))

            self.statusbar.showMessage('表格所示为训练结果')
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_kmeans(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.kmeans_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_kmeans(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 测试结果
    def testresult_kmeans(self):
        try:
            self.labels_kmeans = self.kmeans_model.extract_kmeans_samples(self.test_data)
            self.labels_kmeans = np.mat(self.labels_kmeans).T
            nrows, ncols = self.labels_kmeans.shape
            self.kmeans_ceshijieguo.setRowCount(nrows)
            self.kmeans_ceshijieguo.setColumnCount(ncols)

            for i in range(nrows):
                for j in range(ncols):
                    self.kmeans_ceshijieguo.setItem(i, j, QTableWidgetItem(str(self.labels_kmeans[i, j])))

            self.statusbar.showMessage('表格所示为测试结果')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_kmeans(self):
        try:
            nrows, ncols = self.labels_kmeans.shape
            workbook_kmeans = xlwt.Workbook()
            sheet_kmeans = workbook_kmeans.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_kmeans.write(i, j, float(self.labels_kmeans[i, j]))

            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_kmeans.save(datafile + '.xls')
            self.statusbar.showMessage('聚类结果已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # GMM
    ## 训练模型
    def train_gmm(self):
        try:
            n_components_min = int("".join(filter(str.isdigit, self.n_components_min_gmm.text())))
            n_components_max = int("".join(filter(str.isdigit, self.n_components_max_gmm.text())))
            nrows, ncols = self.train_data.shape
            if (0 < n_components_min <= n_components_max < nrows):
                covariance_type = self.comboBox_covariance_type_gmm.currentText()
                if (covariance_type == 'auto'):
                    cv_types = ['spherical', 'tied', 'diag', 'full']
                else:
                    cv_types = [covariance_type]
                n_components_range = range(n_components_min, n_components_max+1)
                lowest_bic = np.infty
                bic = []
                for cv_type in cv_types:
                    for n_components in n_components_range:
                        gmm = GaussianMixtureModels(self.train_data, n_components=n_components, covariance_type=cv_type,
                                                    preprocess=self.Pre_Processing_gmm.isChecked())
                        gmm.construct_gmm_model()
                        bic.append(gmm.bic_gmm_model())
                        if bic[-1] < lowest_bic:
                            lowest_bic = bic[-1]
                            self.best_gmm = gmm

                self.GMM_model = self.best_gmm
                cp = self.GMM_model.n_components
                cv = self.GMM_model.covariance_type
                self.statusbar.showMessage('模型已训练，选择的模型参量为: '
                                           'n_components='+str(cp)+'，covariance_type='+cv)
                # 画图
                bic = np.array(bic)
                color_iter = ['navy', 'turquoise', 'cornflowerblue', 'darkorange']
                bars = []
                self.fig_gmm.clear()
                plt = self.fig_gmm.add_subplot(111)
                for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
                    xpos = np.array(n_components_range) + .2*i
                    bars.append(plt.bar(xpos, bic[i*len(n_components_range):
                                                  (i+1)*len(n_components_range)],
                                        width=.2, color=color))
                plt.set_xticks(n_components_range)
                plt.set_title('BIC score per model')
                plt.set_xlabel('Number of components')
                plt.legend([b[0] for b in bars], cv_types, loc='upper right')
                self.canvas_gmm.draw()
                self.gmm_beiyesi.setScene(self.graphicscene_gmm)
                self.gmm_beiyesi.show()
            else:
                QMessageBox.information(self, 'Warning', '请设定正确的聚类个数范围', QMessageBox.Ok)
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    ## 训练结果
    def trainresult_gmm(self):
        try:
            self.labels_gmm = self.GMM_model.extract_gmm_samples(self.train_data)
            self.labels_gmm = np.mat(self.labels_gmm).T
            nrows, ncols = self.labels_gmm.shape
            self.gmm_juleijieguo.setRowCount(nrows)
            self.gmm_juleijieguo.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.gmm_juleijieguo.setItem(i, j, QTableWidgetItem(str(self.labels_gmm[i, j])))

            self.statusbar.showMessage('表格所示为训练结果')
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_gmm(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.GMM_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_gmm(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 测试结果
    def testresult_gmm(self):
        try:
            self.labels_gmm = self.GMM_model.extract_gmm_samples(self.test_data)
            self.labels_gmm = np.mat(self.labels_gmm).T
            nrows, ncols = self.labels_gmm.shape
            self.gmm_juleijieguo.setRowCount(nrows)
            self.gmm_juleijieguo.setColumnCount(ncols)

            for i in range(nrows):
                for j in range(ncols):
                    self.gmm_juleijieguo.setItem(i, j, QTableWidgetItem(str(self.labels_gmm[i, j])))

            self.statusbar.showMessage('表格所示为测试结果')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_gmm(self):
        try:
            nrows, ncols = self.labels_gmm.shape
            workbook_gmm = xlwt.Workbook()
            sheet_gmm = workbook_gmm.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_gmm.write(i, j, float(self.labels_gmm[i, j]))

            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_gmm.save(datafile + '.xls')
            self.statusbar.showMessage('聚类结果已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # HCC
    ## 训练模型
    def train_hcc(self):
        try:
            self.criterion_type = self.comboBox_criterion_hcc.currentText()
            self.criterion_valu = float(self.line_criterion_hcc.text())
            self.linkage_hcc = self.comboBox_linkage_hcc.currentText()
            self.HCC_model = HierarchicalClustering(self.train_data, self.criterion_valu, self.linkage_hcc,
                                                    self.criterion_type, self.Pre_Processing_hcc.isChecked())
            self.HCC_model.construct_hcc_model()
            self.hcc_z = self.HCC_model.construct_hcc_model()
            self.statusbar.showMessage('模型已训练')
            # 画图
            self.fig_hcc.clear()
            plt = self.fig_hcc.add_subplot(111)
            plt.set_title('Hierarchical Clustering Dendrogram')
            plt.set_xlabel('Number of points in node')
            dendrogram(self.hcc_z, ax=plt)
            self.canvas_hcc.draw()
            self.hcc_shuzhuangtu.setScene(self.graphicscene_hcc)
            self.hcc_shuzhuangtu.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    ## 训练结果
    def trainresult_hcc(self):
        try:
            self.labels_hcc = self.HCC_model.func_hcc_labels()
            self.labels_hcc = np.mat(self.labels_hcc).T
            nrows, ncols = self.labels_hcc.shape
            self.hcc_juleijieguo.setRowCount(nrows)
            self.hcc_juleijieguo.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.hcc_juleijieguo.setItem(i, j, QTableWidgetItem(str(self.labels_hcc[i, j])))

            self.statusbar.showMessage('表格所示为训练结果')
        except:
            QMessageBox.information(self, 'Warning', '训练数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_hcc(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.HCC_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_hcc(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 测试结果
    def testresult_hcc(self):
        try:
            self.labels_hcc = self.HCC_model.extract_hcc_samples(self.test_data)
            self.labels_hcc = np.mat(self.labels_hcc)
            nrows, ncols = self.labels_hcc.shape
            self.hcc_juleijieguo.setRowCount(nrows)
            self.hcc_juleijieguo.setColumnCount(ncols)

            for i in range(nrows):
                for j in range(ncols):
                    self.hcc_juleijieguo.setItem(i, j, QTableWidgetItem(str(self.labels_hcc[i, j])))

            self.statusbar.showMessage('表格所示为测试结果')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_hcc(self):
        try:
            nrows, ncols = self.labels_hcc.shape
            workbook_hcc = xlwt.Workbook()
            sheet_hcc = workbook_hcc.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_hcc.write(i, j, float(self.labels_hcc[i, j]))

            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_hcc.save(datafile + '.xls')
            self.statusbar.showMessage('聚类结果已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = JL_window()
    win.show()
    sys.exit(app.exec_())


