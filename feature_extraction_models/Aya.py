import sys
import xlrd
import xlwt
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QGraphicsScene
from win import Ui_MainWindow

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
        self.xunlianmoxing_lda.clicked.connect(self.construct_lda_model)
        self.baocunmoxing_lda.clicked.connect(self.savemodel_lda)
        self.daorumoxing_lda.clicked.connect(self.loadmodel_lda)
        self.ceshi_lda.clicked.connect(self.test_lda)
        self.baocunshuju_lda.clicked.connect(self.savedata_lda)

        #画图
        self.fig_lda = Figure((12, 5))  # 15, 8
        self.canvas_lda = FigureCanvas(self.fig_lda)
        self.graphicscene_lda = QGraphicsScene()
        self.graphicscene_lda.addWidget(self.canvas_lda)
        self.toolbar_lda = NavigationToolbar(self.canvas_lda, self.lda_tu)

        #SOM
        self.xunlianmoxing_som.clicked.connect(self.construct_som_model)
        self.baocunmoxing_som.clicked.connect(self.savemodel_som)
        self.daorumoxing_som.clicked.connect(self.loadmodel_som)
        self.ceshi_som.clicked.connect(self.test_som)
        self.baocunshuju_som.clicked.connect(self.savedata_som)

        # 画图
        self.fig_som = Figure((12, 5))  # 15, 8
        self.canvas_som = FigureCanvas(self.fig_som)
        self.graphicscene_som = QGraphicsScene()
        self.graphicscene_som.addWidget(self.canvas_som)
        self.toolbar_som = NavigationToolbar(self.canvas_som, self.som_tu)

        #t-SNE
        self.xunlianmoxing_tsne.clicked.connect(self.construct_tsne_model)
        self.baocunmoxing_tsne.clicked.connect(self.savemodel_tsne)
        self.daorumoxing_tsne.clicked.connect(self.loadmodel_tsne)
        self.ceshi_tsne.clicked.connect(self.test_tsne)
        self.baocunshuju_tsne.clicked.connect(self.savedata_tsne)

        # 画图
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FS_window()
    win.show()
    sys.exit(app.exec_())