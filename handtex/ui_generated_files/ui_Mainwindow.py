# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (QApplication, QFormLayout, QFrame, QGraphicsView,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSpacerItem, QSpinBox, QSplitter, QStackedWidget,
    QVBoxLayout, QWidget)

from handtex.sketchpad import Sketchpad

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(900, 606)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_3 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 6, 0, 0)
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.splitter.setHandleWidth(3)
        self.splitter.setChildrenCollapsible(False)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 6, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, -1, -1, 1)
        self.pushButton_clear = QPushButton(self.layoutWidget)
        self.pushButton_clear.setObjectName(u"pushButton_clear")
        self.pushButton_clear.setEnabled(False)
        icon = QIcon()
        iconThemeName = u"edit-clear"
        if QIcon.hasThemeIcon(iconThemeName):
            icon = QIcon.fromTheme(iconThemeName)
        else:
            icon.addFile(u".", QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_clear.setIcon(icon)
        self.pushButton_clear.setFlat(True)

        self.horizontalLayout.addWidget(self.pushButton_clear)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.pushButton_undo = QPushButton(self.layoutWidget)
        self.pushButton_undo.setObjectName(u"pushButton_undo")
        self.pushButton_undo.setEnabled(False)
        icon1 = QIcon()
        iconThemeName = u"edit-undo"
        if QIcon.hasThemeIcon(iconThemeName):
            icon1 = QIcon.fromTheme(iconThemeName)
        else:
            icon1.addFile(u".", QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_undo.setIcon(icon1)
        self.pushButton_undo.setFlat(True)

        self.horizontalLayout.addWidget(self.pushButton_undo)

        self.pushButton_redo = QPushButton(self.layoutWidget)
        self.pushButton_redo.setObjectName(u"pushButton_redo")
        self.pushButton_redo.setEnabled(False)
        icon2 = QIcon()
        iconThemeName = u"edit-redo"
        if QIcon.hasThemeIcon(iconThemeName):
            icon2 = QIcon.fromTheme(iconThemeName)
        else:
            icon2.addFile(u".", QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_redo.setIcon(icon2)
        self.pushButton_redo.setFlat(True)

        self.horizontalLayout.addWidget(self.pushButton_redo)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.sketchpad = Sketchpad(self.layoutWidget)
        self.sketchpad.setObjectName(u"sketchpad")
        self.sketchpad.setMinimumSize(QSize(200, 200))
        self.sketchpad.setFrameShape(QFrame.NoFrame)
        self.sketchpad.setFrameShadow(QFrame.Plain)
        self.sketchpad.setLineWidth(0)
        self.sketchpad.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sketchpad.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sketchpad.setRenderHints(QPainter.Antialiasing)
        self.sketchpad.setTransformationAnchor(QGraphicsView.NoAnchor)

        self.verticalLayout.addWidget(self.sketchpad)

        self.verticalLayout.setStretch(1, 1)
        self.splitter.addWidget(self.layoutWidget)
        self.layoutWidget1 = QWidget(self.splitter)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(6, 0, 0, 0)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.pushButton_symbol_list = QPushButton(self.layoutWidget1)
        self.pushButton_symbol_list.setObjectName(u"pushButton_symbol_list")
        icon3 = QIcon()
        iconThemeName = u"search"
        if QIcon.hasThemeIcon(iconThemeName):
            icon3 = QIcon.fromTheme(iconThemeName)
        else:
            icon3.addFile(u".", QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_symbol_list.setIcon(icon3)
        self.pushButton_symbol_list.setFlat(True)

        self.horizontalLayout_2.addWidget(self.pushButton_symbol_list)

        self.pushButton_menu = QPushButton(self.layoutWidget1)
        self.pushButton_menu.setObjectName(u"pushButton_menu")
        icon4 = QIcon()
        iconThemeName = u"application-menu"
        if QIcon.hasThemeIcon(iconThemeName):
            icon4 = QIcon.fromTheme(iconThemeName)
        else:
            icon4.addFile(u".", QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_menu.setIcon(icon4)
        self.pushButton_menu.setFlat(True)

        self.horizontalLayout_2.addWidget(self.pushButton_menu)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.stackedWidget = QStackedWidget(self.layoutWidget1)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.page_classify = QWidget()
        self.page_classify.setObjectName(u"page_classify")
        self.verticalLayout_4 = QVBoxLayout(self.page_classify)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_predictions = QScrollArea(self.page_classify)
        self.scrollArea_predictions.setObjectName(u"scrollArea_predictions")
        self.scrollArea_predictions.setWidgetResizable(True)
        self.widget_predictions = QWidget()
        self.widget_predictions.setObjectName(u"widget_predictions")
        self.widget_predictions.setGeometry(QRect(0, 0, 521, 556))
        self.verticalLayout_7 = QVBoxLayout(self.widget_predictions)
        self.verticalLayout_7.setSpacing(12)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_predictions.setWidget(self.widget_predictions)

        self.verticalLayout_4.addWidget(self.scrollArea_predictions)

        self.stackedWidget.addWidget(self.page_classify)
        self.page_train = QWidget()
        self.page_train.setObjectName(u"page_train")
        self.verticalLayout_5 = QVBoxLayout(self.page_train)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(-1, -1, -1, 12)
        self.label_training_name = QLabel(self.page_train)
        self.label_training_name.setObjectName(u"label_training_name")
        self.label_training_name.setText(u"<Symbol name>")

        self.horizontalLayout_5.addWidget(self.label_training_name)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_3)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label_5 = QLabel(self.page_train)
        self.label_5.setObjectName(u"label_5")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_5)

        self.label_symbol_samples = QLabel(self.page_train)
        self.label_symbol_samples.setObjectName(u"label_symbol_samples")
        self.label_symbol_samples.setText(u"<samples>")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.label_symbol_samples)

        self.label = QLabel(self.page_train)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label)

        self.label_symbol_rarity = QLabel(self.page_train)
        self.label_symbol_rarity.setObjectName(u"label_symbol_rarity")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.label_symbol_rarity)


        self.horizontalLayout_5.addLayout(self.formLayout)


        self.verticalLayout_5.addLayout(self.horizontalLayout_5)

        self.widget_training_symbol = QSvgWidget(self.page_train)
        self.widget_training_symbol.setObjectName(u"widget_training_symbol")
        self.widget_training_symbol.setMinimumSize(QSize(200, 200))

        self.verticalLayout_5.addWidget(self.widget_training_symbol, 0, Qt.AlignHCenter)

        self.verticalSpacer_3 = QSpacerItem(20, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_3)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_6 = QLabel(self.page_train)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_6.addWidget(self.label_6)

        self.label_submission_number = QLabel(self.page_train)
        self.label_submission_number.setObjectName(u"label_submission_number")
        self.label_submission_number.setText(u"<submission>")

        self.horizontalLayout_6.addWidget(self.label_submission_number)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_4)

        self.label_8 = QLabel(self.page_train)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_6.addWidget(self.label_8)

        self.spinBox_max_submissions = QSpinBox(self.page_train)
        self.spinBox_max_submissions.setObjectName(u"spinBox_max_submissions")
        self.spinBox_max_submissions.setMinimum(1)
        self.spinBox_max_submissions.setMaximum(100)

        self.horizontalLayout_6.addWidget(self.spinBox_max_submissions)


        self.verticalLayout_5.addLayout(self.horizontalLayout_6)

        self.label_2 = QLabel(self.page_train)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_5.addWidget(self.label_2, 0, Qt.AlignHCenter)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(-1, -1, -1, 0)
        self.label_3 = QLabel(self.page_train)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_3.addWidget(self.label_3)

        self.horizontalSlider_selection_bias = QSlider(self.page_train)
        self.horizontalSlider_selection_bias.setObjectName(u"horizontalSlider_selection_bias")
        self.horizontalSlider_selection_bias.setMaximum(100)
        self.horizontalSlider_selection_bias.setValue(25)
        self.horizontalSlider_selection_bias.setOrientation(Qt.Horizontal)
        self.horizontalSlider_selection_bias.setTickPosition(QSlider.TicksBelow)
        self.horizontalSlider_selection_bias.setTickInterval(50)

        self.horizontalLayout_3.addWidget(self.horizontalSlider_selection_bias)

        self.label_4 = QLabel(self.page_train)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_3.addWidget(self.label_4)


        self.verticalLayout_5.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(-1, -1, -1, 24)
        self.label_7 = QLabel(self.page_train)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_7.addWidget(self.label_7)

        self.lineEdit_train_symbol = QLineEdit(self.page_train)
        self.lineEdit_train_symbol.setObjectName(u"lineEdit_train_symbol")

        self.horizontalLayout_7.addWidget(self.lineEdit_train_symbol)


        self.verticalLayout_5.addLayout(self.horizontalLayout_7)

        self.pushButton_submit = QPushButton(self.page_train)
        self.pushButton_submit.setObjectName(u"pushButton_submit")
        icon5 = QIcon()
        iconThemeName = u"dialog-ok"
        if QIcon.hasThemeIcon(iconThemeName):
            icon5 = QIcon.fromTheme(iconThemeName)
        else:
            icon5.addFile(u".", QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_submit.setIcon(icon5)

        self.verticalLayout_5.addWidget(self.pushButton_submit)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(-1, 24, -1, -1)
        self.pushButton_skip = QPushButton(self.page_train)
        self.pushButton_skip.setObjectName(u"pushButton_skip")
        icon6 = QIcon()
        iconThemeName = u"media-skip-forward"
        if QIcon.hasThemeIcon(iconThemeName):
            icon6 = QIcon.fromTheme(iconThemeName)
        else:
            icon6.addFile(u".", QSize(), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_skip.setIcon(icon6)

        self.horizontalLayout_4.addWidget(self.pushButton_skip)

        self.pushButton_undo_submit = QPushButton(self.page_train)
        self.pushButton_undo_submit.setObjectName(u"pushButton_undo_submit")
        self.pushButton_undo_submit.setEnabled(False)
        self.pushButton_undo_submit.setIcon(icon1)

        self.horizontalLayout_4.addWidget(self.pushButton_undo_submit)


        self.verticalLayout_5.addLayout(self.horizontalLayout_4)

        self.stackedWidget.addWidget(self.page_train)

        self.verticalLayout_2.addWidget(self.stackedWidget)

        self.splitter.addWidget(self.layoutWidget1)

        self.verticalLayout_3.addWidget(self.splitter)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.pushButton_clear.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
#if QT_CONFIG(tooltip)
        self.pushButton_undo.setToolTip(QCoreApplication.translate("MainWindow", u"Undo", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_undo.setText(QCoreApplication.translate("MainWindow", u"Undo", None))
#if QT_CONFIG(tooltip)
        self.pushButton_redo.setToolTip(QCoreApplication.translate("MainWindow", u"Redo", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_redo.setText(QCoreApplication.translate("MainWindow", u"Redo", None))
        self.pushButton_symbol_list.setText(QCoreApplication.translate("MainWindow", u"Symbol List", None))
        self.pushButton_menu.setText("")
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Samples:", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Rarity:", None))
        self.label_symbol_rarity.setText(QCoreApplication.translate("MainWindow", u"<rarity>", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Submission:", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Submissions per symbol:", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Selection Bias", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Random", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Rare Symbols", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Train specific symbol:", None))
        self.pushButton_submit.setText(QCoreApplication.translate("MainWindow", u"Submit", None))
        self.pushButton_skip.setText(QCoreApplication.translate("MainWindow", u"Skip", None))
        self.pushButton_undo_submit.setText(QCoreApplication.translate("MainWindow", u"Undo Submission", None))
    # retranslateUi

