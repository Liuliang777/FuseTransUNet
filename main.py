import sys
from PyQt5.QtCore import *
from PyQt5 import QtCore
from PyQt5 import QtWidgets,QtGui,QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
#from log import Ui_MainWindow
from ui import Ui_MainWindow as main_window
#from infer import inference

class main_win(QMainWindow, main_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 设置界面

        self.selbtn.clicked.connect(self.readimg)
        self.runbtn.clicked.connect(self.pre)
    def readimg(self):
        try:
            self.p = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", "",
                                                      "jpg (*.jpg);png (*.png);All files (*.*)")[0]  # 起始路径
            if self.p:
                self.filepath = self.p
                self.lineEdit.setText(self.filepath)

                pixmap1 = QPixmap(self.filepath)
                self.imglb1.setPixmap(pixmap1)
                self.imglb1.setScaledContents(True)

        except:
            return 1
    def pre(self):

        d = r'{}'.format(self.p)
        if 'train' in d:
            t = d.replace('train','train_labels')
            pixmap1 = QPixmap(t)
            self.imglb3.setPixmap(pixmap1)
            self.imglb3.setScaledContents(True)
        elif 'test' in d:
            t=d.replace('test','test_labels')
            pixmap1 = QPixmap(t)
            self.imglb3.setPixmap(pixmap1)
            self.imglb3.setScaledContents(True)
        elif 'val' in d:
            t=d.replace('val','val_labels')
            pixmap1 = QPixmap(t)
            self.imglb3.setPixmap(pixmap1)
            self.imglb3.setScaledContents(True)




if __name__ == "__main__":

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)


    main_w = main_win()

    main_w.show()
    sys.exit(app.exec_())