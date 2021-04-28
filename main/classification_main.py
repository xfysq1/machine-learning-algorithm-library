# -*- coding: utf-8 -*-

import sys
import xlrd
import xlwt
import numpy as np
import joblib
import re

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QGraphicsScene
from window.classification_Window import Ui_MainWindow

from classification_models.linear_discriminant_analysis import LinearDiscriminant
from classification_models.support_vector_machines import SupportVectorClassification
from classification_models.decision_tree import DecisionTree
from classification_models.random_forest import RandomForest
from classification_models.multi_layer_perceptron import MultiLayerPerceptron
from classification_models.naive_bayesian_model import NaiveBayesian

class FL_window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):

        super(FL_window, self).__init__(parent)
        self.setupUi(self)

        self.setWindowTitle('智能分类')  # 给窗口设置标题
        self.setStyleSheet("#Main_Window{background-color: white}")
        self.setStyleSheet("#stackedWidget{background-color: white}")

        self.traindata.triggered.connect(self.load_traindata)
        self.testdata.triggered.connect(self.load_testdata)

        self.button_lda.clicked.connect(self.topage_lda)
        self.button_svm.clicked.connect(self.topage_svm)
        self.button_dt.clicked.connect(self.topage_dt)
        self.button_rf.clicked.connect(self.topage_rf)
        self.button_ann.clicked.connect(self.topage_ann)
        self.button_nbc.clicked.connect(self.topage_nbc)

        # LDA
        self.xunlian_lda.clicked.connect(self.train_lda)
        self.baocunmoxing_lda.clicked.connect(self.savemodel_lda)
        self.daorumoxing_lda.clicked.connect(self.loadmodel_lda)
        self.ceshijieguo_lda.clicked.connect(self.testresult_lda)
        self.baocunshuju_lda.clicked.connect(self.savedata_lda)

        # SVM
        self.xunlian_svm.clicked.connect(self.train_svm)
        self.baocunmoxing_svm.clicked.connect(self.savemodel_svm)
        self.daorumoxing_svm.clicked.connect(self.loadmodel_svm)
        self.ceshijieguo_svm.clicked.connect(self.testresult_svm)
        self.baocunshuju_svm.clicked.connect(self.savedata_svm)

        # DT
        self.xunlian_dt.clicked.connect(self.train_dt)
        self.baocunmoxing_dt.clicked.connect(self.savemodel_dt)
        self.daorumoxing_dt.clicked.connect(self.loadmodel_dt)
        self.ceshijieguo_dt.clicked.connect(self.testresult_dt)
        self.baocunshuju_dt.clicked.connect(self.savedata_dt)
        '''
        # 画图
        self.fig_dt = Figure(figsize=(10, 6))
        self.canvas_dt = FigureCanvas(self.fig_dt)
        self.graphicscene_dt = QGraphicsScene()
        self.graphicscene_dt.addWidget(self.canvas_dt)
        self.toolbar_dt = NavigationToolbar(self.canvas_dt, self.dt_shujiegou)
        '''
        # 最大深度
        self.maxdep_dt.setVisible(False)  # 隐藏
        self.checkBox_maxdep_dt.stateChanged.connect(self.change_line_maxdep_dt)

        # RF
        self.xunlian_rf.clicked.connect(self.train_rf)
        self.baocunmoxing_rf.clicked.connect(self.savemodel_rf)
        self.daorumoxing_rf.clicked.connect(self.loadmodel_rf)
        self.ceshijieguo_rf.clicked.connect(self.testresult_rf)
        self.baocunshuju_rf.clicked.connect(self.savedata_rf)
        # 特征数量
        self.numfea_rf.setVisible(False)
        self.checkBox_numfea_rf.stateChanged.connect(self.change_line_numfea_rf)

        # ANN
        self.xunlian_ann.clicked.connect(self.train_ann)
        self.baocunmoxing_ann.clicked.connect(self.savemodel_ann)
        self.daorumoxing_ann.clicked.connect(self.loadmodel_ann)
        self.ceshijieguo_ann.clicked.connect(self.testresult_ann)
        self.baocunshuju_ann.clicked.connect(self.savedata_ann)

        # NBC
        self.xunlian_nbc.clicked.connect(self.train_nbc)
        self.baocunmoxing_nbc.clicked.connect(self.savemodel_nbc)
        self.daorumoxing_nbc.clicked.connect(self.loadmodel_nbc)
        self.ceshijieguo_nbc.clicked.connect(self.testresult_nbc)
        self.baocunshuju_nbc.clicked.connect(self.savedata_nbc)
        # 类型
        self.label_alpha_nbc.setVisible(False)
        self.alpha_nbc.setVisible(False)
        self.label_binarize_nbc.setVisible(False)
        self.binarize_nbc.setVisible(False)
        self.Pre_Processing_nbc.setVisible(False)
        self.comboBox_type_nbc.currentIndexChanged.connect(self.change_line_nbc)


    # 界面切换
    def topage_lda(self):
        self.stackedWidget.setCurrentWidget(self.page_LDA)
    def topage_svm(self):
        self.stackedWidget.setCurrentWidget(self.page_SVM)
    def topage_dt(self):
        self.stackedWidget.setCurrentWidget(self.page_DT)
    def topage_rf(self):
        self.stackedWidget.setCurrentWidget(self.page_RF)
    def topage_ann(self):
        self.stackedWidget.setCurrentWidget(self.page_ANN)
    def topage_nbc(self):
        self.stackedWidget.setCurrentWidget(self.page_NBC)

    def change_line_maxdep_dt(self):
        if self.checkBox_maxdep_dt.isChecked():
            self.maxdep_dt.setVisible(True)
        else:
            self.maxdep_dt.setVisible(False)

    def change_line_numfea_rf(self):
        if self.checkBox_numfea_rf.isChecked():
            self.numfea_rf.setVisible(True)
        else:
            self.numfea_rf.setVisible(False)

    def change_line_nbc(self):
        combo = self.comboBox_type_nbc.currentText()
        if (combo == 'GaussianNB'):
            self.label_alpha_nbc.setVisible(False)
            self.alpha_nbc.setVisible(False)
            self.label_binarize_nbc.setVisible(False)
            self.binarize_nbc.setVisible(False)
            self.Pre_Processing_nbc.setVisible(False)
        else:
            self.label_alpha_nbc.setVisible(True)
            self.alpha_nbc.setVisible(True)
            if (combo == 'BernoulliNB'):
                self.label_binarize_nbc.setVisible(True)
                self.binarize_nbc.setVisible(True)
                self.Pre_Processing_nbc.setVisible(True)
            else:
                self.label_binarize_nbc.setVisible(False)
                self.binarize_nbc.setVisible(False)
                self.Pre_Processing_nbc.setVisible(False)


    # 导入训练数据
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
                    self.trainWidget.setItem(i, j, QTableWidgetItem(str(table.cell_value(i, j))))
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

    # LDA
    # 训练模型
    def train_lda(self):
        try:
            self.solver = self.comboBox_solver_lda.currentText()
            nrows, ncols = self.train_data.shape
            # train_data的最后一列为标签y，y为整数
            self.trainx_lda = self.train_data[:, 0:(ncols-1)]
            self.trainy_lda = self.train_data[:, (ncols-1)]
            self.lda_model = LinearDiscriminant(self.trainx_lda, self.trainy_lda, self.solver,
                                                self.Pre_Processing_lda.isChecked())
            self.lda_model.construct_lda_model()
            self.statusbar.showMessage('模型已训练')
            '''
            # 显示权重
            weight_lda = self.lda_model.weight_lda_model()
            nrows0, ncols0 = weight_lda.shape
            self.lda_weight.setRowCount(nrows0)
            self.lda_weight.setColumnCount(ncols0)
            for i in range(nrows0):
                for j in range(ncols0):
                    self.lda_weight.setItem(i, j, QTableWidgetItem(str(weight_lda[i, j])))
            self.labels_lda = weight_lda
            '''
            # 显示分类指标
            self.lda_clfReport.setText(self.lda_model.clfReport())
            self.lda_model.construct_lda_model()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错或标签没有为整数', QMessageBox.Ok)

    # 保存模型
    def savemodel_lda(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.lda_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    # 导入模型
    def loadmodel_lda(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    # 测试结果
    def testresult_lda(self):
        try:
            self.labels_lda = self.lda_model.extract_lda_samples(self.test_data)
            self.labels_lda = np.mat(self.labels_lda).T
            nrows, ncols = self.labels_lda.shape
            self.lda_fenleijieguo.setRowCount(nrows)
            self.lda_fenleijieguo.setColumnCount(ncols)

            for i in range(nrows):
                for j in range(ncols):
                    self.lda_fenleijieguo.setItem(i, j, QTableWidgetItem(str(self.labels_lda[i, j])))

            self.statusbar.showMessage('表格所示为测试结果')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    # 保存数据
    def savedata_lda(self):
        try:
            nrows, ncols = self.labels_lda.shape
            workbook_lda = xlwt.Workbook()
            sheet_lda = workbook_lda.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_lda.write(i, j, float(self.labels_lda[i, j]))

            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_lda.save(datafile + '.xls')
            self.statusbar.showMessage('分类结果已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # SVM
    # 训练模型
    def train_svm(self):
        try:
            self.svm_c = float(self.c_svm.text())
            self.svm_kernel = self.comboBox_kernel_svm.currentText()
            self.svm_decision = self.comboBox_decision_svm.currentText()
            nrows, ncols = self.train_data.shape
            # train_data的最后一列为标签y，y为整数
            self.trainx_svm = self.train_data[:, 0:(ncols-1)]
            self.trainy_svm = self.train_data[:, (ncols-1)]
            self.svm_model = SupportVectorClassification(self.trainx_svm, self.trainy_svm, C=self.svm_c,
                                                         kernel=self.svm_kernel,
                                                         class_weight=self.balanced_svm.isChecked(),
                                                         decision=self.svm_decision,
                                                         preprocess=self.Pre_Processing_svm.isChecked())
            self.svm_model.construct_svc_model()
            self.statusbar.showMessage('模型已训练')
            '''
            # 显示支持向量索引
            support = self.svm_model.support_svc_model()
            support = np.mat(support).T
            nrows0, ncols0 = support.shape
            self.svm_support.setRowCount(nrows0)
            self.svm_support.setColumnCount(ncols0)
            for i in range(nrows0):
                for j in range(ncols0):
                    self.svm_support.setItem(i, j, QTableWidgetItem(str(support[i, j])))
            self.labels_svm = support
            '''
            # 显示分类指标
            self.svm_clfReport.setText(self.svm_model.clfReport())
            self.svm_model.construct_svc_model()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错或标签没有为整数', QMessageBox.Ok)

    # 保存模型
    def savemodel_svm(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.svm_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    # 导入模型
    def loadmodel_svm(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    # 测试结果
    def testresult_svm(self):
        try:
            self.labels_svm = self.svm_model.extract_svc_samples(self.test_data)
            self.labels_svm = np.mat(self.labels_svm).T
            nrows, ncols = self.labels_svm.shape
            self.svm_fenleijieguo.setRowCount(nrows)
            self.svm_fenleijieguo.setColumnCount(ncols)

            for i in range(nrows):
                for j in range(ncols):
                    self.svm_fenleijieguo.setItem(i, j, QTableWidgetItem(str(self.labels_svm[i, j])))

            self.statusbar.showMessage('表格所示为测试结果')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    # 保存数据
    def savedata_svm(self):
        try:
            nrows, ncols = self.labels_svm.shape
            workbook_svm = xlwt.Workbook()
            sheet_svm = workbook_svm.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_svm.write(i, j, float(self.labels_svm[i, j]))

            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_svm.save(datafile + '.xls')
            self.statusbar.showMessage('分类结果已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # DT
    # 训练模型
    def train_dt(self):
        try:
            # numy = int(self.numy_dt.text())
            numy = 1
            split = int(self.minspl_dt.text())
            leaf = int(self.minlea_dt.text())
            if self.checkBox_maxdep_dt.isChecked():
                depth = int(self.maxdep_dt.text())
            else:
                depth = None
            nrows, ncols = self.train_data.shape
            # train_data的最后numy列为标签y1,y2...,y为整数
            self.trainx_dt = self.train_data[:, 0:(ncols-numy)]
            self.trainy_dt = self.train_data[:, (ncols-numy):]
            self.dt_model = DecisionTree(self.trainx_dt, self.trainy_dt, depth,
                                         split, leaf, self.checkBox_balanced_dt.isChecked())
            self.dt_model.construct_dt_model()
            self.statusbar.showMessage('模型已训练')
            '''
            # 画树结构图
            self.fig_dt.clear()
            plt = self.fig_dt.add_subplot(111)
            self.dt_model.tree_dt_model(plt)
            self.canvas_dt.draw()
            self.dt_shujiegou.setScene(self.graphicscene_dt)
            self.dt_shujiegou.show()
            '''
            # 显示分类指标
            self.dt_clfReport.setText(self.dt_model.clfReport())
            self.dt_model.construct_dt_model()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错或标签没有为整数', QMessageBox.Ok)

    # 保存模型
    def savemodel_dt(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.dt_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    # 导入模型
    def loadmodel_dt(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    # 测试结果
    def testresult_dt(self):
        try:
            self.labels_dt = self.dt_model.extract_dt_samples(self.test_data)
            self.labels_dt = np.mat(self.labels_dt)
            nrows, ncols = self.labels_dt.shape
            if nrows == 1:
                self.labels_dt = np.mat(self.labels_dt).T
            nrows, ncols = self.labels_dt.shape
            self.dt_fenleijieguo.setRowCount(nrows)
            self.dt_fenleijieguo.setColumnCount(ncols)

            for i in range(nrows):
                for j in range(ncols):
                    self.dt_fenleijieguo.setItem(i, j, QTableWidgetItem(str(self.labels_dt[i, j])))

            self.statusbar.showMessage('表格所示为测试结果')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    # 保存数据
    def savedata_dt(self):
        try:
            nrows, ncols = self.labels_dt.shape
            workbook_dt = xlwt.Workbook()
            sheet_dt = workbook_dt.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_dt.write(i, j, float(self.labels_dt[i, j]))

            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_dt.save(datafile + '.xls')
            self.statusbar.showMessage('分类结果已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # RF
    # 训练模型
    def train_rf(self):
        try:
            # ny_rf = int(self.numy_rf.text())
            ny_rf = 1
            ntree_rf = int(self.numtree_rf.text())
            weight_rf = self.comboBox_weight_rf.currentText()
            if self.checkBox_numfea_rf.isChecked():
                nfea = int(self.numfea_rf.text())
            else:
                nfea = 'auto'
            nrows, ncols = self.train_data.shape
            # train_data的最后ny_rf列为标签y1,y2...,y为整数
            self.trainx_rf = self.train_data[:, 0:(ncols-ny_rf)]
            self.trainy_rf = self.train_data[:, (ncols-ny_rf):]
            if ny_rf == 1:
                self.trainy_rf = self.trainy_rf.ravel()  # 将多维数组拉平为一维数组，不然有warning
            self.rf_model = RandomForest(self.trainx_rf, self.trainy_rf, ntree_rf, nfea, weight_rf)
            self.rf_model.construct_rf_model()
            self.statusbar.showMessage('模型已训练')
            '''
            # 显示特征重要性
            feaimp = self.rf_model.feaimportance_rf_model()
            feaimp = np.mat(feaimp)
            nrows0, ncols0 = feaimp.shape
            self.rf_feaimportance.setRowCount(nrows0)
            self.rf_feaimportance.setColumnCount(ncols0)
            for i in range(nrows0):
                for j in range(ncols0):
                    self.rf_feaimportance.setItem(i, j, QTableWidgetItem(str(feaimp[i, j])))
            self.labels_rf = feaimp
            '''
            # 显示分类指标
            self.rf_clfReport.setText(self.rf_model.clfReport())
            self.rf_model.construct_rf_model()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错或标签没有为整数', QMessageBox.Ok)

    # 保存模型
    def savemodel_rf(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.rf_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    # 导入模型
    def loadmodel_rf(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    # 测试结果
    def testresult_rf(self):
        try:
            self.labels_rf = self.rf_model.extract_rf_samples(self.test_data)
            self.labels_rf = np.mat(self.labels_rf)
            nrows, ncols = self.labels_rf.shape
            if nrows == 1:
                self.labels_rf = np.mat(self.labels_rf).T
            nrows, ncols = self.labels_rf.shape
            self.rf_fenleijieguo.setRowCount(nrows)
            self.rf_fenleijieguo.setColumnCount(ncols)

            for i in range(nrows):
                for j in range(ncols):
                    self.rf_fenleijieguo.setItem(i, j, QTableWidgetItem(str(self.labels_rf[i, j])))

            self.statusbar.showMessage('表格所示为测试结果')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    # 保存数据
    def savedata_rf(self):
        try:
            nrows, ncols = self.labels_rf.shape
            workbook_rf = xlwt.Workbook()
            sheet_rf = workbook_rf.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_rf.write(i, j, float(self.labels_rf[i, j]))

            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_rf.save(datafile + '.xls')
            self.statusbar.showMessage('分类结果已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # ANN
    # 训练模型
    def train_ann(self):
        try:
            hidden = []
            for i in re.split(';|,|；|，', self.hidden_ann.text()):
                hidden.append(int("".join(filter(str.isdigit, i))))
            hidden = tuple(hidden)
            active = self.comboBox_active_ann.currentText()
            solver = self.comboBox_solver_ann.currentText()
            a = float(self.alpha_ann.text())
            nrows, ncols = self.train_data.shape
            # train_data的最后一列为标签y
            self.trainx_ann = self.train_data[:, 0:(ncols-1)]
            self.trainy_ann = self.train_data[:, (ncols-1)]
            self.ann_model = MultiLayerPerceptron(self.trainx_ann, self.trainy_ann, hidden,
                                                  active, solver, a)
            self.ann_model.construct_mlp_model()
            self.statusbar.showMessage('模型已训练，网络总层数='+str(self.ann_model.mlp.n_layers_))
            # 显示分类指标
            self.ann_clfReport.setText(self.ann_model.clfReport())
            self.ann_model.construct_mlp_model()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错或标签没有为整数', QMessageBox.Ok)

    # 保存模型
    def savemodel_ann(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.ann_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    # 导入模型
    def loadmodel_ann(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    # 测试结果
    def testresult_ann(self):
        try:
            self.labels_ann = self.ann_model.extract_mlp_samples(self.test_data)
            self.labels_ann = np.mat(self.labels_ann).T
            nrows, ncols = self.labels_ann.shape
            self.ann_fenleijieguo.setRowCount(nrows)
            self.ann_fenleijieguo.setColumnCount(ncols)

            for i in range(nrows):
                for j in range(ncols):
                    self.ann_fenleijieguo.setItem(i, j, QTableWidgetItem(str(self.labels_ann[i, j])))

            self.statusbar.showMessage('表格所示为测试结果')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    # 保存数据
    def savedata_ann(self):
        try:
            nrows, ncols = self.labels_ann.shape
            workbook_ann = xlwt.Workbook()
            sheet_ann = workbook_ann.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_ann.write(i, j, float(self.labels_ann[i, j]))

            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_ann.save(datafile + '.xls')
            self.statusbar.showMessage('分类结果已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # NBC
    # 训练模型
    def train_nbc(self):
        try:
            type_nbc = self.comboBox_type_nbc.currentText()
            a = float(self.alpha_nbc.text())
            bina = float(self.binarize_nbc.text())
            nrows, ncols = self.train_data.shape
            # train_data的最后一列为标签y
            self.trainx_nbc = self.train_data[:, 0:(ncols-1)]
            self.trainy_nbc = self.train_data[:, (ncols-1)]
            self.nbc_model = NaiveBayesian(self.trainx_nbc, self.trainy_nbc, type_nbc, a, bina,
                                           self.Pre_Processing_nbc.isChecked())
            self.nbc_model.construct_nbc_model()
            self.statusbar.showMessage('模型已训练')
            # 显示分类指标
            self.nbc_clfReport.setText(self.nbc_model.clfReport())
            self.nbc_model.construct_nbc_model()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错或标签没有为整数', QMessageBox.Ok)

    # 保存模型
    def savemodel_nbc(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.nbc_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    # 导入模型
    def loadmodel_nbc(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    # 测试结果
    def testresult_nbc(self):
        try:
            self.labels_nbc = self.nbc_model.extract_nbc_samples(self.test_data)
            self.labels_nbc = np.mat(self.labels_nbc).T
            nrows, ncols = self.labels_nbc.shape
            self.nbc_fenleijieguo.setRowCount(nrows)
            self.nbc_fenleijieguo.setColumnCount(ncols)

            for i in range(nrows):
                for j in range(ncols):
                    self.nbc_fenleijieguo.setItem(i, j, QTableWidgetItem(str(self.labels_nbc[i, j])))

            self.statusbar.showMessage('表格所示为测试结果')
        except:
            QMessageBox.information(self, 'Warning', '测试数据未导入或模型未定义', QMessageBox.Ok)

    # 保存数据
    def savedata_nbc(self):
        try:
            nrows, ncols = self.labels_nbc.shape
            workbook_nbc = xlwt.Workbook()
            sheet_nbc = workbook_nbc.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_nbc.write(i, j, float(self.labels_nbc[i, j]))

            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_nbc.save(datafile + '.xls')
            self.statusbar.showMessage('分类结果已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FL_window()
    win.show()
    sys.exit(app.exec_())


