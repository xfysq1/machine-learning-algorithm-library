import sys
import xlrd
import xlwt
import numpy as np
from sklearn import preprocessing
import joblib

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QGraphicsScene
from window.intelligent_optimization_Window import Ui_MainWindow

from intelligent_optimization_models.io_GA import io_GA as io_GA
from intelligent_optimization_models.io_DE import io_DE as io_DE
from intelligent_optimization_models.io_PSO import io_PSO as io_PSO
from intelligent_optimization_models.aim_func import aim_func_GA,aim_func_DE,aim_func_PSO


class IO_window(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(IO_window, self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle('智能优化')
        self.setStyleSheet("#Main_Window{background-color: white}")
        self.setStyleSheet("#stackedWidget{background-color: white}")

        self.definite_aim_func.triggered.connect(self.open_definite_aim_func)


        # 模型界面选择
        self.button_GA.clicked.connect(self.topage_GA)
        self.button_DE.clicked.connect(self.topage_DE)
        self.button_PSO.clicked.connect(self.topage_PSO)

        # GA
        self.xunlianmoxing_GA.clicked.connect(self.construct_GA_model)
        self.youhuajieguo_GA.clicked.connect(self.extact_GA_result)
        self.baocunmoxing_GA.clicked.connect(self.savemodel_GA)
        self.daorumoxing_GA.clicked.connect(self.loadmodel_GA)
        self.baocunshuju_GA.clicked.connect(self.savedata_GA)
        ## 画图
        self.fig_GA = Figure((12, 5))  # 15, 8
        self.canvas_GA = FigureCanvas(self.fig_GA)
        self.graphicscene_GA = QGraphicsScene()
        self.graphicscene_GA.addWidget(self.canvas_GA)
        self.toolbar_GA = NavigationToolbar(self.canvas_GA, self.youhuajieguotu_GA)

        # DE
        self.xunlianmoxing_DE.clicked.connect(self.construct_DE_model)
        self.youhuajieguo_DE.clicked.connect(self.extact_DE_result)
        self.baocunmoxing_DE.clicked.connect(self.savemodel_DE)
        self.daorumoxing_DE.clicked.connect(self.loadmodel_DE)
        self.baocunshuju_DE.clicked.connect(self.savedata_DE)
        ## 画图
        self.fig_DE = Figure((12, 5))  # 15, 8
        self.canvas_DE = FigureCanvas(self.fig_DE)
        #self.canvas_pca.setParent(self.pca_gongxiantu)
        self.graphicscene_DE = QGraphicsScene()
        self.graphicscene_DE.addWidget(self.canvas_DE)
        self.toolbar_DE = NavigationToolbar(self.canvas_DE, self.youhuajieguotu_DE)

        # PSO
        self.xunlianmoxing_PSO.clicked.connect(self.construct_PSO_model)
        self.youhuajieguo_PSO.clicked.connect(self.extact_PSO_result)
        self.baocunmoxing_PSO.clicked.connect(self.savemodel_PSO)
        self.daorumoxing_PSO.clicked.connect(self.loadmodel_PSO)
        self.baocunshuju_PSO.clicked.connect(self.savedata_PSO)
        ## 画图
        self.fig_PSO = Figure((12, 5))  # 15, 8
        self.canvas_PSO = FigureCanvas(self.fig_PSO)
        self.graphicscene_PSO = QGraphicsScene()
        self.graphicscene_PSO.addWidget(self.canvas_PSO)
        self.toolbar_PSO = NavigationToolbar(self.canvas_PSO, self.youhuajieguotu_PSO)



    # 界面切换
    def topage_GA(self):
        self.stackedWidget.setCurrentWidget(self.page_GA)
    def topage_DE(self):
        self.stackedWidget.setCurrentWidget(self.page_DE)
    def topage_PSO(self):
        self.stackedWidget.setCurrentWidget(self.page_PSO)


    # 导入训练数据
    def open_definite_aim_func(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "定义目标函数及约束条件")
        except:
            QMessageBox.information(self, 'Warning', '请定义目标函数及约束条件', QMessageBox.Ok)



    # GA
    ## 训练模型
    def construct_GA_model(self):
        try:
            #获取输入值
            number_of_variable = int("".join(filter(str.isdigit, self.number_of_variables_GA.text())))
            NIND = int("".join(filter(str.isdigit, self.NIND_GA.text())))
            MAXGEN = int("".join(filter(str.isdigit, self.MAXGEN_GA.text())))
            lower_bounds = []
            for i in self.lower_bounds_GA.text().split(","):
                lower_bounds.append(float("".join(filter(str.isascii, i))))
            # lower_bounds = [-1,-1]
            upper_bounds = []
            for i in self.upper_bounds_GA.text().split(","):
                upper_bounds.append(float("".join(filter(str.isascii, i))))
            # upper_bounds = [1, 1]
            precision = float("".join(filter(str.isascii, self.precision_GA.text())))
            # precision = 1e-7

            self.GA_model = io_GA(func=aim_func_GA.aim_func,n_dim=number_of_variable,size_pop=NIND,
                                  max_iter=MAXGEN,lb=lower_bounds,ub=upper_bounds,precision=precision,
                                  constraint_eq=aim_func_GA.constraint_eq, constraint_ueq=aim_func_GA.constraint_ueq)
            self.GA_model.construct_GA_model()
            obj_trace_GA = self.GA_model.generation_best_Y
            self.statusbar.showMessage('模型已训练')
            len_obj_trace_GA = np.array(obj_trace_GA).shape[0]
            index = np.arange(len_obj_trace_GA)
            self.fig_GA.clear()
            plt = self.fig_GA.add_subplot(111)
            plt.plot(index, obj_trace_GA, '-')
            self.canvas_GA.draw()
            self.youhuajieguotu_GA.setScene(self.graphicscene_GA)
            self.youhuajieguotu_GA.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)


    # 训练特征
    def extact_GA_result(self):
        try:
            self.result_GA = self.GA_model.construct_GA_model()
            # nrows, ncols = self.features_pca.shape
            nrows = 2
            ncols = 1

            self.youhuajieguobiao_GA.setRowCount(nrows)
            self.youhuajieguobiao_GA.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.youhuajieguobiao_GA.setItem(i, j, QTableWidgetItem(str(self.result_GA[i])))
            self.statusbar.showMessage('表格所示为优化结果')
        except:
            QMessageBox.information(self, 'Warning', '模型未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_GA(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.GA_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_GA(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_GA(self):
        try:
            # nrows, ncols = self.features_pca.shape
            nrows = 4
            ncols = 1
            workbook_GA = xlwt.Workbook()
            sheet_GA = workbook_GA.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_GA.write(i, j, float(self.features_GA[i]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_GA.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # DE
    ## 训练模型
    def construct_DE_model(self):
        try:
            #获取输入值
            number_of_variable = int("".join(filter(str.isdigit, self.number_of_variables_DE.text())))
            NIND = int("".join(filter(str.isdigit, self.NIND_DE.text())))
            MAXGEN = int("".join(filter(str.isdigit, self.MAXGEN_DE.text())))
            lower_bounds = []
            for i in self.lower_bounds_DE.text().split(","):
                lower_bounds.append(float("".join(filter(str.isascii, i))))
            # lower_bounds = [0, 0, 0]
            upper_bounds = []
            for i in self.upper_bounds_DE.text().split(","):
                upper_bounds.append(float("".join(filter(str.isascii, i))))
            # upper_bounds = [5, 5, 5]

            self.DE_model = io_DE(func=aim_func_DE.aim_func_DE, n_dim=number_of_variable,size_pop=NIND,
                                  max_iter=MAXGEN,lb=lower_bounds, ub=upper_bounds,
                                  constraint_eq=aim_func_DE.constraint_eq, constraint_ueq=aim_func_DE.constraint_ueq)
            print(1)
            self.DE_model.construct_DE_model()
            print(2)
            obj_trace_DE = self.DE_model.generation_best_Y
            print(3)
            self.statusbar.showMessage('模型已训练')
            len_obj_trace_DE = np.array(obj_trace_DE).shape[0]
            index = np.arange(len_obj_trace_DE)
            self.fig_DE.clear()
            plt = self.fig_DE.add_subplot(111)
            plt.plot(index, obj_trace_DE, '-')
            self.canvas_DE.draw()
            self.youhuajieguotu_DE.setScene(self.graphicscene_DE)
            self.youhuajieguotu_DE.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)


    # 训练特征
    def extact_DE_result(self):
        try:
            self.result_DE = self.DE_model.construct_DE_model()
            # nrows, ncols = self.features_pca.shape
            nrows = 2
            ncols = 1

            self.youhuajieguobiao_DE.setRowCount(nrows)
            self.youhuajieguobiao_DE.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    self.youhuajieguobiao_DE.setItem(i, j, QTableWidgetItem(str(self.result_DE[i])))
            self.statusbar.showMessage('表格所示为优化结果')
        except:
            QMessageBox.information(self, 'Warning', '模型未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_DE(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.DE_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_DE(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_DE(self):
        try:
            # nrows, ncols = self.features_pca.shape
            nrows = 2
            ncols = 1
            workbook_DE = xlwt.Workbook()
            sheet_DE = workbook_DE.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_DE.write(i, j, float(self.features_DE[i]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_DE.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)

    # PSO
    ## 训练模型
    def construct_PSO_model(self):
        try:
            #获取输入值
            number_of_variable = int("".join(filter(str.isdigit, self.number_of_variables_PSO.text())))
            NIND = int("".join(filter(str.isdigit, self.NIND_PSO.text())))
            MAXGEN = int("".join(filter(str.isdigit, self.MAXGEN_PSO.text())))
            lower_bounds = []
            for i in self.lower_bounds_PSO.text().split(","):
                lower_bounds.append(float("".join(filter(str.isascii, i))))
            # lower_bounds = [0, -1, 0.5]
            upper_bounds = []
            for i in self.upper_bounds_PSO.text().split(","):
                upper_bounds.append(float("".join(filter(str.isascii, i))))
            # upper_bounds = [1, 1, 1]
            inertia = float("".join(filter(str.isascii, self.inertia_PSO.text())))
            personal_best_parameter = float("".join(filter(str.isascii, self.personal_best_parameter_PSO.text())))
            global_best_parameter = float("".join(filter(str.isascii, self.global_best_parameter_PSO.text())))

            self.PSO_model = io_PSO(func=aim_func_PSO.aim_func,dim=number_of_variable,pop=NIND,
                                    max_iter=MAXGEN,lb=lower_bounds,ub=upper_bounds,
                                    w=inertia,c1=personal_best_parameter,c2=global_best_parameter)
            self.PSO_model.construct_PSO_model()
            obj_trace_PSO = self.PSO_model.gbest_y_hist
            self.statusbar.showMessage('模型已训练')
            len_obj_trace_PSO = np.array(obj_trace_PSO).shape[0]
            index = np.arange(len_obj_trace_PSO)
            self.fig_PSO.clear()
            plt = self.fig_PSO.add_subplot(111)
            plt.plot(index, obj_trace_PSO, '-')
            self.canvas_PSO.draw()
            self.youhuajieguotu_PSO.setScene(self.graphicscene_PSO)
            self.youhuajieguotu_PSO.show()
        except:
            QMessageBox.information(self, 'Warning', '模型参数出错', QMessageBox.Ok)


    # 训练特征
    def extact_PSO_result(self):
        try:
            self.result_PSO = self.PSO_model.construct_PSO_model()
            # nrows, ncols = self.features_pca.shape
            nrows = 2
            ncols = 1

            self.youhuajieguobiao_PSO.setRowCount(nrows)
            self.youhuajieguobiao_PSO.setColumnCount(ncols)
            for i in range(nrows):
                for j in range(ncols):
                    print(i)
                    print(j)
                    self.youhuajieguobiao_PSO.setItem(i, j, QTableWidgetItem(str(self.result_PSO[i])))
            self.statusbar.showMessage('表格所示为优化结果')
        except:
            QMessageBox.information(self, 'Warning', '模型未导入或模型未定义', QMessageBox.Ok)

    ## 保存模型
    def savemodel_PSO(self):
        datafile, _ = QFileDialog.getSaveFileName(self, "保存模型")
        try:
            joblib.dump(self.PSO_model, datafile + '.m')
            self.statusbar.showMessage('模型已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用模型', QMessageBox.Ok)

    ## 导入模型
    def loadmodel_PSO(self):
        try:
            datafile, _ = QFileDialog.getOpenFileName(self, "导入模型")
            joblib.load(datafile)
            self.statusbar.showMessage('模型已加载')
        except:
            QMessageBox.information(self, 'Warning', '模型未定义', QMessageBox.Ok)

    ## 保存数据
    def savedata_PSO(self):
        try:
            # nrows, ncols = self.features_pca.shape
            nrows = 2
            ncols = 1
            workbook_PSO = xlwt.Workbook()
            sheet_PSO = workbook_PSO.add_sheet('sheet1')

            for i in range(nrows):
                for j in range(ncols):
                    sheet_PSO.write(i, j, float(self.features_PSO[i]))
            datafile, _ = QFileDialog.getSaveFileName(self, "保存至")
            workbook_PSO.save(datafile + '.xls')
            self.statusbar.showMessage('特征数据已保存')
        except:
            QMessageBox.information(self, 'Warning', '无可用数据', QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = IO_window()
    win.show()
    sys.exit(app.exec_())
