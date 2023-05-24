import sys
import clean_module as cm
import os
from PyQt5 import uic, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QLineEdit, QComboBox, QLabel, QSpinBox, QPushButton, QMessageBox


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("diploma.ui", self)

    @pyqtSlot(name='on_cleanButton_clicked')
    def clean_image(self):
        flag = 0
        if self.ui.salt.isChecked():
            cm.COLOR = cm.WHITE
            flag = 1
        elif self.ui.pepper.isChecked():
            cm.COLOR = cm.BLACK
            flag = 2

        if flag == 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setWindowTitle("Что-то пошло не так")
            msgBox.setText("Пожалуйста, укажите тип шума")
            msgBox.exec()
            return

        path = self.file_text.toPlainText()
        if os.path.isfile(path):
            cm.clean_image(path, flag)
            return

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setWindowTitle("Что-то пошло не так")
        msgBox.setText("Укажите верный путь до файла c изображением")
        msgBox.exec()

    @pyqtSlot(name='on_compare_button_clicked')
    def count_metric(self):
        flag = 0
        if self.ui.salt.isChecked():
            cm.COLOR = cm.WHITE
            flag = 1
        elif self.ui.pepper.isChecked():
            cm.COLOR = cm.BLACK
            flag = 2

        if flag == 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setWindowTitle("Что-то пошло не так")
            msgBox.setText("Пожалуйста, укажите тип шума")
            msgBox.exec()
            return

        image_path = self.file_text.toPlainText()
        src_path = self.src_text.toPlainText()
        if os.path.isfile(src_path) and os.path.isfile(image_path):
            cm.check_metrics(image_path, src_path, flag)
            return

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setWindowTitle("Что-то пошло не так")
        msgBox.setText("Укажите верный путь до файла c изображениями")
        msgBox.exec()

    @pyqtSlot(name='on_diffButton_clicked')
    def compare_methods(self):
        flag = 0
        if self.ui.salt.isChecked():
            cm.COLOR = cm.WHITE
            flag = 1
        elif self.ui.pepper.isChecked():
            cm.COLOR = cm.BLACK
            flag = 2

        if flag == 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setWindowTitle("Что-то пошло не так")
            msgBox.setText("Пожалуйста, укажите тип шума")
            msgBox.exec()
            return

        path = self.file_text.toPlainText()
        if os.path.isfile(path):
            cm.compare(path, flag)
            return

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setWindowTitle("Что-то пошло не так")
        msgBox.setText("Укажите верный путь до файла c изображением")
        msgBox.exec()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
