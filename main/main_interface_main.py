import sys
import xlrd
import xlwt
import numpy as np
from sklearn import preprocessing
import joblib

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QGraphicsScene,QDialog
from window.main_interface_Window import Ui_MainWindow

from main.feature_selection_main import FS_window as feature_selection_page
from main.feature_extraction_main import FS_window as feature_extraction_page
from main.space_projection_main import SP_window as space_projection_page
from main.cluster_main import JL_window as cluster_page
from main.classification_main import FL_window as classification_page
from main.intelligent_optimization_main import IO_window as inteligence_optimization_page

class main_interface_window(QMainWindow, Ui_MainWindow):

    def __init__(self):

        super(main_interface_window, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('人工智能软件')
        self.setStyleSheet("#Main_Window{background-color: white}")
        self.setStyleSheet("#stackedWidget{background-color: white}")

        # 模型界面选择
        self.button_feature_selection.clicked.connect(self.jump_to_feature_selection)
        self.button_feature_extraction.clicked.connect(self.jump_to_feature_extraction)
        self.button_space_projection.clicked.connect(self.jump_to_space_projection)
        self.button_cluster.clicked.connect(self.jump_to_cluster)
        self.button_classification.clicked.connect(self.jump_to_classification)
        self.button_intelligence_optimization.clicked.connect(self.jump_to_intelligence_optimization)

    #界面跳转函数
    def jump_to_feature_selection(self):
        self.to_feature_selection = feature_selection_page()
        self.to_feature_selection.show()
    def jump_to_feature_extraction(self):
        self.to_feature_extraction = feature_extraction_page()
        self.to_feature_extraction.show()
    def jump_to_space_projection(self):
        self.to_space_projection = space_projection_page()
        self.to_space_projection.show()
    def jump_to_cluster(self):
        self.to_cluster = cluster_page()
        self.to_cluster.show()
    def jump_to_classification(self):
        self.to_classification = classification_page()
        self.to_classification.show()
    def jump_to_intelligence_optimization(self):
        self.to_inteligence_optimization = inteligence_optimization_page()
        self.to_inteligence_optimization.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = main_interface_window()
    win.show()
    sys.exit(app.exec_())