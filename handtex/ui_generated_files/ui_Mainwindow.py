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
from PySide6.QtWidgets import (QApplication, QGraphicsView, QHBoxLayout, QLabel,
    QListWidgetItem, QMainWindow, QPushButton, QSizePolicy,
    QSlider, QSpacerItem, QSplitter, QStackedWidget,
    QVBoxLayout, QWidget)

from handtex.CustomQ.CListWidget import CListWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_3 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.widget = QWidget(self.splitter)
        self.widget.setObjectName(u"widget")
        self.verticalLayout = QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 6, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.pushButton_clear = QPushButton(self.widget)
        self.pushButton_clear.setObjectName(u"pushButton_clear")
        icon = QIcon(QIcon.fromTheme(u"edit-clear"))
        self.pushButton_clear.setIcon(icon)
        self.pushButton_clear.setFlat(True)

        self.horizontalLayout.addWidget(self.pushButton_clear)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.pushButton_undo = QPushButton(self.widget)
        self.pushButton_undo.setObjectName(u"pushButton_undo")
        icon1 = QIcon(QIcon.fromTheme(u"edit-undo"))
        self.pushButton_undo.setIcon(icon1)
        self.pushButton_undo.setFlat(True)

        self.horizontalLayout.addWidget(self.pushButton_undo)

        self.pushButton_redo = QPushButton(self.widget)
        self.pushButton_redo.setObjectName(u"pushButton_redo")
        icon2 = QIcon(QIcon.fromTheme(u"edit-redo"))
        self.pushButton_redo.setIcon(icon2)
        self.pushButton_redo.setFlat(True)

        self.horizontalLayout.addWidget(self.pushButton_redo)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.verticalSpacer = QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.graphicsView = QGraphicsView(self.widget)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setMinimumSize(QSize(200, 200))

        self.verticalLayout.addWidget(self.graphicsView)

        self.verticalSpacer_2 = QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_2)

        self.verticalLayout.setStretch(2, 1)
        self.splitter.addWidget(self.widget)
        self.widget1 = QWidget(self.splitter)
        self.widget1.setObjectName(u"widget1")
        self.verticalLayout_2 = QVBoxLayout(self.widget1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(6, 0, 0, 0)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.pushButton_menu = QPushButton(self.widget1)
        self.pushButton_menu.setObjectName(u"pushButton_menu")
        icon3 = QIcon(QIcon.fromTheme(u"application-menu"))
        self.pushButton_menu.setIcon(icon3)
        self.pushButton_menu.setFlat(True)

        self.horizontalLayout_2.addWidget(self.pushButton_menu)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.stackedWidget = QStackedWidget(self.widget1)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.page_classify = QWidget()
        self.page_classify.setObjectName(u"page_classify")
        self.verticalLayout_4 = QVBoxLayout(self.page_classify)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.listWidget = CListWidget(self.page_classify)
        self.listWidget.setObjectName(u"listWidget")

        self.verticalLayout_4.addWidget(self.listWidget)

        self.stackedWidget.addWidget(self.page_classify)
        self.page_train = QWidget()
        self.page_train.setObjectName(u"page_train")
        self.verticalLayout_5 = QVBoxLayout(self.page_train)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_training_name = QLabel(self.page_train)
        self.label_training_name.setObjectName(u"label_training_name")
        self.label_training_name.setText(u"<Symbol name>")

        self.horizontalLayout_5.addWidget(self.label_training_name)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_3)

        self.label_symbol_rarity = QLabel(self.page_train)
        self.label_symbol_rarity.setObjectName(u"label_symbol_rarity")

        self.horizontalLayout_5.addWidget(self.label_symbol_rarity)


        self.verticalLayout_5.addLayout(self.horizontalLayout_5)

        self.widget_training_symbol = QWidget(self.page_train)
        self.widget_training_symbol.setObjectName(u"widget_training_symbol")
        self.widget_training_symbol.setMinimumSize(QSize(200, 200))

        self.verticalLayout_5.addWidget(self.widget_training_symbol)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_3)

        self.label_2 = QLabel(self.page_train)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_5.addWidget(self.label_2, 0, Qt.AlignHCenter)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(-1, -1, -1, 24)
        self.label_3 = QLabel(self.page_train)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_3.addWidget(self.label_3)

        self.horizontalSlider_selection_bias = QSlider(self.page_train)
        self.horizontalSlider_selection_bias.setObjectName(u"horizontalSlider_selection_bias")
        self.horizontalSlider_selection_bias.setMaximum(100)
        self.horizontalSlider_selection_bias.setValue(50)
        self.horizontalSlider_selection_bias.setOrientation(Qt.Horizontal)
        self.horizontalSlider_selection_bias.setTickPosition(QSlider.TicksBelow)
        self.horizontalSlider_selection_bias.setTickInterval(50)

        self.horizontalLayout_3.addWidget(self.horizontalSlider_selection_bias)

        self.label_4 = QLabel(self.page_train)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_3.addWidget(self.label_4)


        self.verticalLayout_5.addLayout(self.horizontalLayout_3)

        self.pushButton_submit = QPushButton(self.page_train)
        self.pushButton_submit.setObjectName(u"pushButton_submit")
        icon4 = QIcon(QIcon.fromTheme(u"dialog-ok"))
        self.pushButton_submit.setIcon(icon4)

        self.verticalLayout_5.addWidget(self.pushButton_submit)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(-1, 24, -1, -1)
        self.pushButton_skip = QPushButton(self.page_train)
        self.pushButton_skip.setObjectName(u"pushButton_skip")
        icon5 = QIcon(QIcon.fromTheme(u"media-skip-forward"))
        self.pushButton_skip.setIcon(icon5)

        self.horizontalLayout_4.addWidget(self.pushButton_skip)

        self.pushButton_go_back = QPushButton(self.page_train)
        self.pushButton_go_back.setObjectName(u"pushButton_go_back")
        self.pushButton_go_back.setIcon(icon1)

        self.horizontalLayout_4.addWidget(self.pushButton_go_back)


        self.verticalLayout_5.addLayout(self.horizontalLayout_4)

        self.stackedWidget.addWidget(self.page_train)

        self.verticalLayout_2.addWidget(self.stackedWidget)

        self.splitter.addWidget(self.widget1)

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
        self.pushButton_undo.setText("")
#if QT_CONFIG(tooltip)
        self.pushButton_redo.setToolTip(QCoreApplication.translate("MainWindow", u"Redo", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_redo.setText("")
        self.pushButton_menu.setText("")
        self.label_symbol_rarity.setText(QCoreApplication.translate("MainWindow", u"<rarity>", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Selection Bias", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Random", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Rare Symbols", None))
        self.pushButton_submit.setText(QCoreApplication.translate("MainWindow", u"Submit", None))
        self.pushButton_skip.setText(QCoreApplication.translate("MainWindow", u"Skip", None))
        self.pushButton_go_back.setText(QCoreApplication.translate("MainWindow", u"Go Back", None))
    # retranslateUi

