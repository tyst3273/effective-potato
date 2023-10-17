
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys


class window(QMainWindow):

    def __init__(self,*args,**kwargs):

        super(window,self).__init__(*args,**kwargs)

        self.setWindowTitle('the eyes are the window to the... eyes')

        label = QLabel('label?')

        label.setAlignment(Qt.AlignCenter)

        self.setCentralWidget(label)



app = QApplication(sys.argv)

window = window()
window.show()

app.exec_()

