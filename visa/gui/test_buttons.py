
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

        toolbar = QToolBar('main bar')
        toolbar.setIconSize(QSize(16,16))
        self.addToolBar(toolbar)
    
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        # button 1
        button_action = QAction(QIcon('lightning.png'),'button 1',self)
        button_action.setStatusTip('your button 1')
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        button_action.setCheckable(True)
        
        button_action.setShortcut( QKeySequence('Ctrl+Enter') )

        toolbar.addAction(button_action)
        toolbar.addSeparator()
        
        file_menu.addAction(button_action)
        file_menu.addSeparator()

        # button 2
        button_action = QAction(QIcon('acorn.png'),'button 2',self)
        button_action.setStatusTip('your button 2')
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        button_action.setCheckable(True)

        toolbar.addAction(button_action)
        toolbar.addSeparator()

        file_submenu = file_menu.addMenu("Submenu")
        file_submenu.addAction(button_action)

        # widget         
        toolbar.addWidget(QLabel("Hello"))
        toolbar.addWidget(QCheckBox())

        self.setStatusBar(QStatusBar(self))

    def onMyToolBarButtonClick(self, s):
        print("click", s)



app = QApplication(sys.argv)

window = window()
window.show()

app.exec_()

