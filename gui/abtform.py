# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'about.ui',
# licensing of 'about.ui' applies.
#
# Created: Sun Mar 10 11:33:29 2019
#      by: pyside2-uic  running on PySide2 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout.addWidget(self.graphicsView)
        self.text = QtWidgets.QTextEdit(Dialog)
        self.text.setReadOnly(True)
        self.text.setObjectName("text")
        self.verticalLayout.addWidget(self.text)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtWidgets.QApplication.translate("Dialog", "Dialog", None, -1))

