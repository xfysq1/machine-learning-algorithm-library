import sys
import numpy as np
import joblib

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QGraphicsScene
from window.login_window import Ui_Dialog

from main.main_interface_main import main_interface_window as main_interface_page

class AD_login_window(QMainWindow, Ui_Dialog):

    def __init__(self, parent=None):
        super(AD_login_window, self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle('登录')
        self.setStyleSheet("#Main_Window{background-color: white}")
        self.setStyleSheet("#stackedWidget{background-color: white}")

        self.button_login.clicked.connect(self.jump_to_main_interface)     #单击按钮跳转

    def jump_to_main_interface(self):

        if self.user_name_text.text() == '123':
            if self.password_text.text() == '123':
                self.close()     #登录窗口关闭
                self.to_main_interface = main_interface_page()
                self.to_main_interface.show()     #显示主界面
            else:
                QMessageBox.information(self, 'Warning', '密码错误', QMessageBox.Ok)
        else:
            QMessageBox.information(self, 'Warning', '用户名错误', QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    diaglog = AD_login_window()
    diaglog.show()
    sys.exit(app.exec_())