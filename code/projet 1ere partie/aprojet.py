# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'approjet.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MOULOUYA(object):
    def setupUi(self, MOULOUYA):
        MOULOUYA.setObjectName("MOULOUYA")
        MOULOUYA.resize(800, 600)
        MOULOUYA.setAnimated(True)
        MOULOUYA.setDocumentMode(False)
        self.centralwidget = QtWidgets.QWidget(MOULOUYA)
        self.centralwidget.setObjectName("centralwidget")
        self.import_2 = QtWidgets.QLabel(self.centralwidget)
        self.import_2.setEnabled(True)
        self.import_2.setGeometry(QtCore.QRect(10, 30, 91, 21))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(9)
        self.import_2.setFont(font)
        self.import_2.setStyleSheet("font: 75 8pt \"MS Shell Dlg 2\";\n"
"text-decoration: underline;")
        self.import_2.setFrameShape(QtWidgets.QFrame.Box)
        self.import_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.import_2.setTextFormat(QtCore.Qt.RichText)
        self.import_2.setScaledContents(False)
        self.import_2.setObjectName("import_2")
        self.groupBox1 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox1.setGeometry(QtCore.QRect(0, 100, 191, 41))
        self.groupBox1.setStyleSheet("font: 75 10pt \"MS Shell Dlg 2\";")
        self.groupBox1.setObjectName("groupBox1")
        self.checkBox = QtWidgets.QCheckBox(self.groupBox1)
        self.checkBox.setGeometry(QtCore.QRect(0, 20, 70, 17))
        self.checkBox.setAcceptDrops(False)
        self.checkBox.setChecked(False)
        self.checkBox.setTristate(False)
        self.checkBox.setObjectName("checkBox")
        self.checkBox_2 = QtWidgets.QCheckBox(self.groupBox1)
        self.checkBox_2.setGeometry(QtCore.QRect(60, 20, 70, 17))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.groupBox1)
        self.checkBox_3.setGeometry(QtCore.QRect(120, 20, 70, 17))
        self.checkBox_3.setObjectName("checkBox_3")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 150, 191, 51))
        self.groupBox_2.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.groupBox_2.setStyleSheet("font: 75 10pt \"MS Shell Dlg 2\";")
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton.setGeometry(QtCore.QRect(10, 20, 82, 17))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_2.setGeometry(QtCore.QRect(80, 20, 82, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 320, 171, 161))
        self.groupBox_4.setObjectName("groupBox_4")
        self.label_3 = QtWidgets.QLabel(self.groupBox_4)
        self.label_3.setGeometry(QtCore.QRect(10, 30, 91, 16))
        self.label_3.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_3.setObjectName("label_3")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEdit.setGeometry(QtCore.QRect(100, 30, 61, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(0, 220, 181, 80))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(10, 20, 171, 16))
        self.label.setFrameShape(QtWidgets.QFrame.Panel)
        self.label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.label_2.setGeometry(QtCore.QRect(10, 50, 91, 16))
        self.label_2.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_2.setObjectName("label_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 500, 121, 16))
        self.label_4.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(170, 500, 101, 16))
        self.label_5.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(290, 500, 101, 16))
        self.label_6.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(420, 500, 121, 16))
        self.label_7.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_7.setObjectName("label_7")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(190, 40, 321, 441))
        self.groupBox.setObjectName("groupBox")
        MOULOUYA.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MOULOUYA)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuFICHIER = QtWidgets.QMenu(self.menubar)
        self.menuFICHIER.setObjectName("menuFICHIER")
        self.menuEDITION = QtWidgets.QMenu(self.menubar)
        self.menuEDITION.setObjectName("menuEDITION")
        self.menuFORMULAIRE = QtWidgets.QMenu(self.menubar)
        self.menuFORMULAIRE.setObjectName("menuFORMULAIRE")
        self.menuAFFICHAGE = QtWidgets.QMenu(self.menubar)
        self.menuAFFICHAGE.setObjectName("menuAFFICHAGE")
        self.menuCONFIGURATION = QtWidgets.QMenu(self.menubar)
        self.menuCONFIGURATION.setObjectName("menuCONFIGURATION")
        self.menuFENETRE = QtWidgets.QMenu(self.menubar)
        self.menuFENETRE.setObjectName("menuFENETRE")
        self.menuAIDE = QtWidgets.QMenu(self.menubar)
        self.menuAIDE.setObjectName("menuAIDE")
        MOULOUYA.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MOULOUYA)
        self.statusbar.setObjectName("statusbar")
        MOULOUYA.setStatusBar(self.statusbar)
        self.actionFORMULAIRE = QtWidgets.QAction(MOULOUYA)
        self.actionFORMULAIRE.setObjectName("actionFORMULAIRE")
        self.actionNouveau = QtWidgets.QAction(MOULOUYA)
        self.actionNouveau.setObjectName("actionNouveau")
        self.actionOuvrir = QtWidgets.QAction(MOULOUYA)
        self.actionOuvrir.setObjectName("actionOuvrir")
        self.actionEnregistrer = QtWidgets.QAction(MOULOUYA)
        self.actionEnregistrer.setObjectName("actionEnregistrer")
        self.actionFermer = QtWidgets.QAction(MOULOUYA)
        self.actionFermer.setObjectName("actionFermer")
        self.actionQuitter = QtWidgets.QAction(MOULOUYA)
        self.actionQuitter.setObjectName("actionQuitter")
        self.menuFICHIER.addAction(self.actionNouveau)
        self.menuFICHIER.addAction(self.actionOuvrir)
        self.menuFICHIER.addAction(self.actionEnregistrer)
        self.menuFICHIER.addAction(self.actionFermer)
        self.menuFICHIER.addAction(self.actionQuitter)
        self.menuEDITION.addAction(self.actionFORMULAIRE)
        self.menubar.addAction(self.menuFICHIER.menuAction())
        self.menubar.addAction(self.menuEDITION.menuAction())
        self.menubar.addAction(self.menuFORMULAIRE.menuAction())
        self.menubar.addAction(self.menuAFFICHAGE.menuAction())
        self.menubar.addAction(self.menuCONFIGURATION.menuAction())
        self.menubar.addAction(self.menuFENETRE.menuAction())
        self.menubar.addAction(self.menuAIDE.menuAction())

        self.retranslateUi(MOULOUYA)
        QtCore.QMetaObject.connectSlotsByName(MOULOUYA)
        
    def retranslateUi(self, MOULOUYA):
        _translate = QtCore.QCoreApplication.translate
        MOULOUYA.setWindowTitle(_translate("MOULOUYA", "MOULOUYA"))
        self.import_2.setText(_translate("MOULOUYA", "Importer données"))
        self.groupBox1.setTitle(_translate("MOULOUYA", "Delai de prédiction"))
        self.checkBox.setText(_translate("MOULOUYA", "1mois"))
        self.checkBox_2.setText(_translate("MOULOUYA", "2mois"))
        self.checkBox_3.setText(_translate("MOULOUYA", "3mois"))
        self.groupBox_2.setTitle(_translate("MOULOUYA", "Variable prédite"))
        self.radioButton.setText(_translate("MOULOUYA", "Classe"))
        self.radioButton_2.setText(_translate("MOULOUYA", "periode de l\'année"))
        self.groupBox_4.setTitle(_translate("MOULOUYA", "Parametres du modele"))
        self.label_3.setText(_translate("MOULOUYA", "Alg aprentissage"))
        self.groupBox_3.setTitle(_translate("MOULOUYA", "Prétraitement"))
        self.label.setText(_translate("MOULOUYA", "Décomposition en ondelettes"))
        self.label_2.setText(_translate("MOULOUYA", "Normalisation"))
        self.label_4.setText(_translate("MOULOUYA", " Apprentissage et test"))
        self.label_5.setText(_translate("MOULOUYA", "toutes les données"))
        self.label_6.setText(_translate("MOULOUYA", "nouvelles données"))
        self.label_7.setText(_translate("MOULOUYA", "Exporter les résultats"))
        self.groupBox.setTitle(_translate("MOULOUYA", "Affichage des résultats"))
        self.menuFICHIER.setTitle(_translate("MOULOUYA", "FICHIER"))
        self.menuEDITION.setTitle(_translate("MOULOUYA", "EDITION"))
        self.menuFORMULAIRE.setTitle(_translate("MOULOUYA", "FORMULAIRE"))
        self.menuAFFICHAGE.setTitle(_translate("MOULOUYA", "AFFICHAGE"))
        self.menuCONFIGURATION.setTitle(_translate("MOULOUYA", "CONFIGURATION"))
        self.menuFENETRE.setTitle(_translate("MOULOUYA", "FENETRE"))
        self.menuAIDE.setTitle(_translate("MOULOUYA", "AIDE"))
        self.actionFORMULAIRE.setText(_translate("MOULOUYA", "FORMULAIRE"))
        self.actionNouveau.setText(_translate("MOULOUYA", "Nouveau"))
        self.actionOuvrir.setText(_translate("MOULOUYA", "Ouvrir"))
        self.actionEnregistrer.setText(_translate("MOULOUYA", "Enregistrer"))
        self.actionFermer.setText(_translate("MOULOUYA", "Fermer "))
        self.actionQuitter.setText(_translate("MOULOUYA", "Quitter"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MOULOUYA = QtWidgets.QMainWindow()
    ui = Ui_MOULOUYA()
    ui.setupUi(MOULOUYA)
    MOULOUYA.show()
    sys.exit(app.exec_())

