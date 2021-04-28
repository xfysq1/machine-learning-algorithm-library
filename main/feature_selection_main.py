import sys
import xlrd
import xlwt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QGraphicsScene
from window.feature_selection_Window import Ui_MainWindow

from feature_selection_models.fs_pearson import fs_pearson as fs_pearson
from feature_selection_models.fs_MI import fs_MI as fs_MI
from feature_selection_models.fs_MRMR import fs_MRMR as fs_MRMR
from feature_selection_models.fs_RIVI import fs_RIVI as fs_RIVI
from feature_selection_models.fs_LASSO import fs_LASSO as fs_LASSO
from feature_selection_models.fs_XGboost import fs_XGboost as fs_XGboost
from feature_selection_models.fs_SNN import fs_SNN as fs_SNN

class FS_window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(FS_window, self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle('特征选择')
        self.setStyleSheet("#Main_Window{background-color: white}")
        self.setStyleSheet("#stackedWidget{background-color: white}")

        self.load_data.triggered.connect(self.load_datafile)

        # 模型界面选择
        self.button_pearson.clicked.connect(self.topage_pearson)
        self.button_MI.clicked.connect(self.topage_MI)
        self.button_MRMR.clicked.connect(self.topage_MRMR)
        self.button_RIVI.clicked.connect(self.topage_RIVI)
        self.button_LASSO.clicked.connect(self.topage_LASSO)
        self.button_XGboost.clicked.connect(self.topage_XGboost)
        self.button_SNN.clicked.connect(self.topage_SNN)

        # pearson
        self.xunlianmoxing_pearson.clicked.connect(self.construct_pearson_model)
        self.xunliantezheng_pearson.clicked.connect(self.extact_pearson_feature)
        self.baocunmoxing_pearson.clicked.connect(self.savemodel_pearson)
        self.daorumoxing_pearson.clicked.connect(self.loadmodel_pearson)
        self.baocunshuju_pearson.clicked.connect(self.savedata_pearson)
        ## 画图
        self.fig_pearson = Figure((12, 5))  # 15, 8
        self.canvas_pearson = FigureCanvas(self.fig_pearson)
        self.graphicscene_pearson = QGraphicsScene()
        self.graphicscene_pearson.addWidget(self.canvas_pearson)
        self.toolbar_pearson = NavigationToolbar(self.canvas_pearson, self.pearson_gongxiantu)
        # MI
        self.xunlianmoxing_MI.clicked.connect(self.construct_MI_model)
        self.xunliantezheng_MI.clicked.connect(self.extact_MI_feature)
        self.baocunmoxing_MI.clicked.connect(self.savemodel_MI)
        self.daorumoxing_MI.clicked.connect(self.loadmodel_MI)
        self.baocunshuju_MI.clicked.connect(self.savedata_MI)
        ## 画图
        self.fig_MI = Figure((12, 5))  # 15, 8
        self.canvas_MI = FigureCanvas(self.fig_MI)
        self.graphicscene_MI = QGraphicsScene()
        self.graphicscene_MI.addWidget(self.canvas_MI)
        self.toolbar_MI = NavigationToolbar(self.canvas_MI, self.MI_gongxiantu)
        # MRMR
        self.xunlianmoxing_MRMR.clicked.connect(self.construct_MRMR_model)
        self.xunliantezheng_MRMR.clicked.connect(self.extact_MRMR_feature)
        self.baocunmoxing_MRMR.clicked.connect(self.savemodel_MRMR)
        self.daorumoxing_MRMR.clicked.connect(self.loadmodel_MRMR)
        self.baocunshuju_MRMR.clicked.connect(self.savedata_MRMR)
        ## 画图
        self.fig_MRMR = Figure((12, 5))  # 15, 8
        self.canvas_MRMR = FigureCanvas(self.fig_MRMR)
        self.graphicscene_MRMR = QGraphicsScene()
        self.graphicscene_MRMR.addWidget(self.canvas_MRMR)
        self.toolbar_MRMR = NavigationToolbar(self.canvas_MRMR, self.MRMR_gongxiantu)
       # RIVI
        self.xunlianmoxing_RIVI.clicked.connect(self.construct_RIVI_model)
        self.xunliantezheng_RIVI.clicked.connect(self.extact_RIVI_feature)
        self.baocunmoxing_RIVI.clicked.connect(self.savemodel_RIVI)
        self.daorumoxing_RIVI.clicked.connect(self.loadmodel_RIVI)
        self.baocunshuju_RIVI.clicked.connect(self.savedata_RIVI)
        ## 画图
        self.fig_RIVI = Figure((12, 5))  # 15, 8
        self.canvas_RIVI = FigureCanvas(self.fig_RIVI)
        self.graphicscene_RIVI = QGraphicsScene()
        self.graphicscene_RIVI.addWidget(self.canvas_RIVI)
        self.toolbar_RIVI = NavigationToolbar(self.canvas_RIVI, self.RIVI_gongxiantu)
        # LASSO
        self.xunlianmoxing_LASSO.clicked.connect(self.construct_LASSO_model)
        self.xunliantezheng_LASSO.clicked.connect(self.extact_LASSO_feature)
        self.baocunmoxing_LASSO.clicked.connect(self.savemodel_LASSO)
        self.daorumoxing_LASSO.clicked.connect(self.loadmodel_LASSO)
        self.baocunshuju_LASSO.clicked.connect(self.savedata_LASSO)
        ## 画图
        self.fig_LASSO = Figure((12, 5))  # 15, 8
        self.canvas_LASSO = FigureCanvas(self.fig_LASSO)
        self.graphicscene_LASSO = QGraphicsScene()
        self.graphicscene_LASSO.addWidget(self.canvas_LASSO)
        self.toolbar_LASSO = NavigationToolbar(self.canvas_LASSO, self.LASSO_gongxiantu)
        # XGboost
        self.xunlianmoxing_XGboost.clicked.connect(self.construct_XGboost_model)
        self.xunliantezheng_XGboost.clicked.connect(self.extact_XGboost_feature)
        self.baocunmoxing_XGboost.clicked.connect(self.savemodel_XGboost)
        self.daorumoxing_XGboost.clicked.connect(self.loadmodel_XGboost)
        self.baocunshuju_XGboost.clicked.connect(self.savedata_XGboost)
        ## 画图
        self.fig_XGboost = Figure((12, 5))  # 15, 8
        self.canvas_XGboost = FigureCanvas(self.fig_XGboost)
        self.graphicscene_XGboost = QGraphicsScene()
        self.graphicscene_XGboost.addWidget(self.canvas_XGboost)
        self.toolbar_XGboost = NavigationToolbar(self.canvas_XGboost, self.XGboost_gongxiantu)
        # SNN
        self.xunlianmoxing_SNN.clicked.connect(self.construct_SNN_model)
        self.xunliantezheng_SNN.clicked.connect(self.extact_SNN_feature)
        self.baocunmoxing_SNN.clicked.connect(self.savemodel_SNN)
        self.daorumoxing_SNN.clicked.connect(self.loadmodel_SNN)
        self.baocunshuju_SNN.clicked.connect(self.savedata_SNN)
        ## 画图
        self.fig_SNN = Figure((12, 5))  # 15, 8
        self.canvas_SNN = FigureCanvas(self.fig_SNN)
        self.graphicscene_SNN = QGraphicsScene()
        self.graphicscene_SNN.addWidget(self.canvas_SNN)
        self.toolbar_SNN = NavigationToolbar(self.canvas_SNN, self.SNN_gongxiantu)

    # 界面切换
    def topage_pearson(self):
        self.stackedWidget.setCurrentWidget(self.page_pearson)
    def topage_MI(self):
        self.stackedWidget.setCurrentWidget(self.page_MI)
    def topage_MRMR(self):
        self.stackedWidget.setCurrentWidget(self.page_MRMR)
    def topage_RIVI(self):
        self.stackedWidget.setCurrentWidget(self.page_RIVI)
    def topage_LASSO(self):
        self.stackedWidget.setCurrentWidget(self.page_LASSO)
    def topage_XGboost(self):
        self.stackedWidget.setCurrentWidget(self.page_XGboost)
    def topage_SNN(self):
        self.stackedWidget.setCurrentWidget(self.page_SNN)

    # 导入训练数据
    def load_datafile(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "选择数据")
            xlrd.Book.encoding = "gbk"
            data = pd.read_excel(datafile)     #读取数据文件
            table = data.values     #提取数据
            self.variable_info = data.columns.values     #提取变量名
            nrows,ncols = table.shape
            self.trainWidget.setRowCount(nrows)
            self.trainWidget.setColumnCount(ncols)
            self.trainWidget.setHorizontalHeaderLabels(self.variable_info)
            self.train_data = np.zeros((nrows, ncols))

            for i in range(nrows):
                for j in range(ncols):
                    self.trainWidget.setItem(i, j, QTableWidgetItem(str(table[i, j])))
                    self.train_data[i, j] = table[i, j]
            self.statusbar.showMessage('数据已导入')
        except:
            QMessageBox.information(self, 'Warning', '数据为EXCEL表格', QMessageBox.Ok)

    # pearson
    ## 训练模型
    def construct_pearson_model(self):
        try:
            n_components = int("".join(filter(str.isdigit, self.n_components_pearson.text())))
            nrows, ncols = self.train_data.shape
            if (n_components > ncols):
                QMessageBox.information(self, 'Warning', '请设定小于原始维度的特征维度', QMessageBox.Ok)
            else:
                self.pearson_model = fs_pearson(self.train_data, n_components=n_components,
                                              preprocess=self.Pre_Processing_pearson.isChecked())     #预处理过程置1则无故障
                score,pearson_index = self.pearson_model.construct_pearson_model()     #得分和变量索引
                self.pearson_variable_name = self.variable_info[pearson_index]     #根据索引提取变量名
                self.statusbar.showMessage('模型已训练')
                index = np.arange(n_components)
                self.fig_pearson.clear()
                plt = self.fig_pearson.add_subplot(111)
                plt.bar(index, score, 0.2)
                self.canvas_pearson.draw()
                self.pearson_gongxiantu.setScene(self.graphicscene_pearson)
                self.pearson_gongxiantu.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    # 训练特征
    def extact_pearson_feature(self):
        try:
            self.features_pearson = self.pearson_model.extract_pearson_feature(self.train_data)
            nrows,ncols = self.features_pearson.shape
#            nrows = 5
#            ncols = 1

            self.pearson_features.setRowCount(nrows)
            self.pearson_features.setColumnCount(ncols+1)
            strs = ["得分", "变量索引", "变量名"]
            self.pearson_features.setHorizontalHeaderLabels(strs)
            for i in range(nrows):
                for j in range(ncols):
                    self.pearson_features.setItem(i,j, QTableWidgetItem(str(self.features_pearson[i,j])))
            for i in range(nrows):
                self.pearson_features.setItem(i,2, QTableWidgetItem(self.pearson_variable_name[i]))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
             QMessageBox.information(self, 'Warning', '数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_pearson(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.pearson_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_pearson(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_pearson(self):
        try:
            nrows, ncols = self.features_pearson.shape
            workbook_pearson = xlwt.Workbook()
            sheet_pearson = workbook_pearson.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_pearson.write(i, j, float(self.features_pearson[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_pearson.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)



    # MI
    ## 训练模型
    def construct_MI_model(self):
        try:
            n_components = int("".join(filter(str.isdigit, self.n_components_MI.text())))
            nrows, ncols = self.train_data.shape
            if (n_components > ncols):
                QMessageBox.information(self, 'Warning', '请设定小于原始维度的特征维度', QMessageBox.Ok)
            else:
                self.MI_model = fs_MI(self.train_data, n_components=n_components,
                                              preprocess=self.Pre_Processing_MI.isChecked())     #预处理过程置1则无故障
                score,MI_index = self.MI_model.construct_MI_model()
                self.MI_variable_name = self.variable_info[MI_index]     #根据索引提取变量名
                self.statusbar.showMessage('模型已训练')
                index = np.arange(n_components)
                self.fig_MI.clear()
                plt = self.fig_MI.add_subplot(111)
                plt.bar(index, score, 0.2)
                self.canvas_MI.draw()
                self.MI_gongxiantu.setScene(self.graphicscene_MI)
                self.MI_gongxiantu.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    # 训练特征
    def extact_MI_feature(self):
        try:
            self.features_MI = self.MI_model.extract_MI_feature(self.train_data)
            nrows,ncols = self.features_MI.shape

            self.MI_features.setRowCount(nrows)
            self.MI_features.setColumnCount(ncols+1)
            strs = ["得分", "变量索引", "变量名"]
            self.MI_features.setHorizontalHeaderLabels(strs)
            for i in range(nrows):
                for j in range(ncols):
                    self.MI_features.setItem(i,j, QTableWidgetItem(str(self.features_MI[i,j])))
            for i in range(nrows):
                self.MI_features.setItem(i,2, QTableWidgetItem(self.MI_variable_name[i]))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
             QMessageBox.information(self, 'Warning', '数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_MI(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.MI_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_MI(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_MI(self):
        try:
            nrows, ncols = self.features_MI.shape
            workbook_MI = xlwt.Workbook()
            sheet_MI = workbook_MI.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_MI.write(i, j, float(self.features_MI[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_MI.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)



    # MRMR
    ## 训练模型
    def construct_MRMR_model(self):
        try:
            n_components = int("".join(filter(str.isdigit, self.n_components_MRMR.text())))
            nrows, ncols = self.train_data.shape
            if (n_components > ncols):
                QMessageBox.information(self, 'Warning', '请设定小于原始维度的特征维度', QMessageBox.Ok)
            else:
                self.MRMR_model = fs_MRMR(self.train_data, n_components=n_components,
                                              preprocess=self.Pre_Processing_MRMR.isChecked())     #预处理过程置1则无故障
                score,MRMR_index = self.MRMR_model.construct_MRMR_model()
                self.MRMR_variable_name = self.variable_info[MRMR_index]     #根据索引提取变量名
                self.statusbar.showMessage('模型已训练')
                index = np.arange(n_components)
                self.fig_MRMR.clear()
                plt = self.fig_MRMR.add_subplot(111)
                plt.bar(index, score, 0.2)
                self.canvas_MRMR.draw()
                self.MRMR_gongxiantu.setScene(self.graphicscene_MRMR)
                self.MRMR_gongxiantu.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    # 训练特征
    def extact_MRMR_feature(self):
        try:
            self.features_MRMR = self.MRMR_model.extract_MRMR_feature(self.train_data)
            nrows,ncols = self.features_MRMR.shape
#            nrows = 5
#            ncols = 1

            self.MRMR_features.setRowCount(nrows)
            self.MRMR_features.setColumnCount(ncols+1)
            strs = ["得分", "变量索引", "变量名"]
            self.MRMR_features.setHorizontalHeaderLabels(strs)
            for i in range(nrows):
                for j in range(ncols):
                    self.MRMR_features.setItem(i,j, QTableWidgetItem(str(self.features_MRMR[i,j])))
            for i in range(nrows):
                self.MRMR_features.setItem(i,2, QTableWidgetItem(self.MRMR_variable_name[i]))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
             QMessageBox.information(self, 'Warning', '数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_MRMR(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.MRMR_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_MRMR(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_MRMR(self):
        try:
            nrows, ncols = self.features_MRMR.shape
            workbook_MRMR = xlwt.Workbook()
            sheet_MRMR = workbook_MRMR.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_MRMR.write(i, j, float(self.features_MRMR[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_MRMR.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)



    # RIVI
    ## 训练模型
    def construct_RIVI_model(self):
        try:
            n_components = int("".join(filter(str.isdigit, self.n_components_RIVI.text())))
            nrows, ncols = self.train_data.shape
            if (n_components > ncols):
                QMessageBox.information(self, 'Warning', '请设定小于原始维度的特征维度', QMessageBox.Ok)
            else:
                self.RIVI_model = fs_RIVI(self.train_data, n_components=n_components,
                                              preprocess=self.Pre_Processing_RIVI.isChecked())     #预处理过程置1则无故障
                score,RIVI_index = self.RIVI_model.construct_RIVI_model()
                self.RIVI_variable_name = self.variable_info[RIVI_index]     #根据索引提取变量名
                self.statusbar.showMessage('模型已训练')
                index = np.arange(n_components)
                self.fig_RIVI.clear()
                plt = self.fig_RIVI.add_subplot(111)
                plt.bar(index, score, 0.2)
                self.canvas_RIVI.draw()
                self.RIVI_gongxiantu.setScene(self.graphicscene_RIVI)
                self.RIVI_gongxiantu.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    # 训练特征
    def extact_RIVI_feature(self):
        try:
            self.features_RIVI = self.RIVI_model.extract_RIVI_feature(self.train_data)
            nrows,ncols = self.features_RIVI.shape
#            nrows = 5
#            ncols = 1

            self.RIVI_features.setRowCount(nrows)
            self.RIVI_features.setColumnCount(ncols+1)
            strs = ["得分", "变量索引", "变量名"]
            self.RIVI_features.setHorizontalHeaderLabels(strs)
            for i in range(nrows):
                for j in range(ncols):
                    self.RIVI_features.setItem(i,j, QTableWidgetItem(str(self.features_RIVI[i,j])))
            for i in range(nrows):
                self.RIVI_features.setItem(i,2, QTableWidgetItem(self.RIVI_variable_name[i]))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
             QMessageBox.information(self, 'Warning', '数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_RIVI(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.RIVI_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_RIVI(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_RIVI(self):
        try:
            nrows, ncols = self.features_RIVI.shape
            workbook_RIVI = xlwt.Workbook()
            sheet_RIVI = workbook_RIVI.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_RIVI.write(i, j, float(self.features_RIVI[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_RIVI.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)



    # LASSO
    ## 训练模型
    def construct_LASSO_model(self):
        try:
            n_components = int("".join(filter(str.isdigit, self.n_components_LASSO.text())))
            nrows, ncols = self.train_data.shape
            if (n_components > ncols):
                QMessageBox.information(self, 'Warning', '请设定小于原始维度的特征维度', QMessageBox.Ok)
            else:
                self.LASSO_model = fs_LASSO(self.train_data, n_components=n_components,
                                              preprocess=self.Pre_Processing_LASSO.isChecked())     #预处理过程置1则无故障
                score,LASSO_index = self.LASSO_model.construct_LASSO_model()     #画图需要一维数组
                self.LASSO_variable_name = self.variable_info[LASSO_index]     #根据索引提取变量名
                self.statusbar.showMessage('模型已训练')
                index = np.arange(n_components)
                self.fig_LASSO.clear()
                plt = self.fig_LASSO.add_subplot(111)
                plt.bar(index, score, 0.2)
                self.canvas_LASSO.draw()
                self.LASSO_gongxiantu.setScene(self.graphicscene_LASSO)
                self.LASSO_gongxiantu.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    # 训练特征
    def extact_LASSO_feature(self):
        try:
            self.features_LASSO = self.LASSO_model.extract_LASSO_feature(self.train_data)
            nrows,ncols = self.features_LASSO.shape
#            nrows = 5
#            ncols = 1

            self.LASSO_features.setRowCount(nrows)
            self.LASSO_features.setColumnCount(ncols+1)
            strs = ["得分", "变量索引", "变量名"]
            self.LASSO_features.setHorizontalHeaderLabels(strs)
            for i in range(nrows):
                for j in range(ncols):
                    self.LASSO_features.setItem(i,j, QTableWidgetItem(str(self.features_LASSO[i,j])))
            for i in range(nrows):
                self.LASSO_features.setItem(i,2, QTableWidgetItem(self.LASSO_variable_name[i]))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
             QMessageBox.information(self, 'Warning', '数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_LASSO(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.LASSO_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_LASSO(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_LASSO(self):
        try:
            nrows, ncols = self.features_LASSO.shape
            workbook_LASSO = xlwt.Workbook()
            sheet_LASSO = workbook_LASSO.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_LASSO.write(i, j, float(self.features_LASSO[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_LASSO.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # XGboost
    ## 训练模型
    def construct_XGboost_model(self):
        try:
            n_components = int("".join(filter(str.isdigit, self.n_components_XGboost.text())))
            nrows, ncols = self.train_data.shape
            if (n_components > ncols):
                QMessageBox.information(self, 'Warning', '请设定小于原始维度的特征维度', QMessageBox.Ok)
            else:
                self.XGboost_model = fs_XGboost(self.train_data, n_components=n_components,
                                                preprocess=self.Pre_Processing_XGboost.isChecked())  # 预处理过程置1则无故障
                score, XGboost_index = self.XGboost_model.construct_XGboost_model()
                self.XGboost_variable_name = self.variable_info[XGboost_index]     #根据索引提取变量名
                self.statusbar.showMessage('模型已训练')
                index = np.arange(n_components)
                self.fig_XGboost.clear()
                plt = self.fig_XGboost.add_subplot(111)
                plt.bar(index, score, 0.2)
                self.canvas_XGboost.draw()
                self.XGboost_gongxiantu.setScene(self.graphicscene_XGboost)
                self.XGboost_gongxiantu.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    # 训练特征
    def extact_XGboost_feature(self):
        try:
            self.features_XGboost = self.XGboost_model.extract_XGboost_feature(self.train_data)
            nrows, ncols = self.features_XGboost.shape
            #            nrows = 5
            #            ncols = 1

            self.XGboost_features.setRowCount(nrows)
            self.XGboost_features.setColumnCount(ncols+1)
            strs = ["得分", "变量索引", "变量名"]
            self.XGboost_features.setHorizontalHeaderLabels(strs)
            for i in range(nrows):
                for j in range(ncols):
                    self.XGboost_features.setItem(i, j, QTableWidgetItem(str(self.features_XGboost[i, j])))
            for i in range(nrows):
                self.XGboost_features.setItem(i,2, QTableWidgetItem(self.XGboost_variable_name[i]))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
            QMessageBox.information(self, 'Warning', '数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_XGboost(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.XGboost_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_XGboost(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_XGboost(self):
        try:
            nrows, ncols = self.features_XGboost.shape
            workbook_XGboost = xlwt.Workbook()
            sheet_XGboost = workbook_XGboost.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_XGboost.write(i, j, float(self.features_XGboost[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_XGboost.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)


    # SNN
    ## 训练模型
    def construct_SNN_model(self):
        try:
            ### 数据标准化
            self.x_sae = self.train_data
            self.StandardScaler_x_sae = preprocessing.StandardScaler().fit(self.x_sae)
            self.x_sae = self.StandardScaler_x_sae.transform(self.x_sae)
            n_components = int("".join(filter(str.isdigit, self.n_components_SNN.text())))
            hidden_dims = int("".join(filter(str.isdigit, self.hidden_dims_SNN.text())))
            epochs = int("".join(filter(str.isdigit, self.epoch_SNN.text())))
            batch_size = int("".join(filter(str.isdigit, self.batch_size_SNN.text())))
            nrows, ncols = self.train_data.shape
            if (n_components > ncols):
                QMessageBox.information(self, 'Warning', '请设定小于原始维度的特征维度', QMessageBox.Ok)
            else:
                self.SNN_model = fs_SNN(self.x_sae, n_components=n_components,
                                                    hidden_dims=hidden_dims,
                                                    epochs = epochs,
                                                    batch_size = batch_size)     #预处理过程置1则无故障
                score,SNN_index = self.SNN_model.construct_SNN_model()
                self.SNN_variable_name = self.variable_info[SNN_index]     #根据索引提取变量名
                self.statusbar.showMessage('模型已训练')
                index = np.arange(n_components)
                self.fig_SNN.clear()
                plt = self.fig_SNN.add_subplot(111)
                plt.bar(index, score, 0.2)
                self.canvas_SNN.draw()
                self.SNN_gongxiantu.setScene(self.graphicscene_SNN)
                self.SNN_gongxiantu.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)

    # 训练特征
    def extact_SNN_feature(self):
        try:
            self.features_SNN = self.SNN_model.extract_SNN_feature()
            nrows,ncols = self.features_SNN.shape
#            nrows = 5
#            ncols = 1

            self.SNN_features.setRowCount(nrows)
            self.SNN_features.setColumnCount(ncols+1)
            strs = ["得分", "变量索引", "变量名"]
            self.SNN_features.setHorizontalHeaderLabels(strs)
            for i in range(nrows):
                for j in range(ncols):
                    self.SNN_features.setItem(i,j, QTableWidgetItem(str(self.features_SNN[i,j])))
            for i in range(nrows):
                self.SNN_features.setItem(i,2, QTableWidgetItem(self.SNN_variable_name[i]))
            self.statusbar.showMessage('表格所示为训练特征')
        except:
             QMessageBox.information(self, 'Warning', '数据未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_SNN(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.SNN_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_SNN(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_SNN(self):
        try:
            nrows, ncols = self.features_SNN.shape
            workbook_SNN = xlwt.Workbook()
            sheet_SNN = workbook_SNN.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_SNN.write(i, j, float(self.features_SNN[i, j]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_SNN.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FS_window()
    win.show()
    sys.exit(app.exec_())