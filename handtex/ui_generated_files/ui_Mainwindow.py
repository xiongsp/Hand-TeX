# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QFormLayout, QFrame,
    QGraphicsView, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSizePolicy, QSlider,
    QSpacerItem, QSpinBox, QSplitter, QStackedWidget,
    QVBoxLayout, QWidget)

from handtex.CustomQ.CScrollArea import CScrollArea
from handtex.sketchpad import Sketchpad

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(900, 680)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_3 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 6, 0, 0)
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(3)
        self.splitter.setChildrenCollapsible(False)
        self.stackedWidget_left = QStackedWidget(self.splitter)
        self.stackedWidget_left.setObjectName(u"stackedWidget_left")
        self.page_sketch = QWidget()
        self.page_sketch.setObjectName(u"page_sketch")
        self.verticalLayout = QVBoxLayout(self.page_sketch)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, -1, -1, 1)
        self.pushButton_clear = QPushButton(self.page_sketch)
        self.pushButton_clear.setObjectName(u"pushButton_clear")
        self.pushButton_clear.setEnabled(False)
        icon = QIcon(QIcon.fromTheme(u"edit-clear"))
        self.pushButton_clear.setIcon(icon)
        self.pushButton_clear.setFlat(True)

        self.horizontalLayout.addWidget(self.pushButton_clear)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.pushButton_undo = QPushButton(self.page_sketch)
        self.pushButton_undo.setObjectName(u"pushButton_undo")
        self.pushButton_undo.setEnabled(False)
        icon1 = QIcon(QIcon.fromTheme(u"edit-undo"))
        self.pushButton_undo.setIcon(icon1)
        self.pushButton_undo.setFlat(True)

        self.horizontalLayout.addWidget(self.pushButton_undo)

        self.pushButton_redo = QPushButton(self.page_sketch)
        self.pushButton_redo.setObjectName(u"pushButton_redo")
        self.pushButton_redo.setEnabled(False)
        icon2 = QIcon(QIcon.fromTheme(u"edit-redo"))
        self.pushButton_redo.setIcon(icon2)
        self.pushButton_redo.setFlat(True)

        self.horizontalLayout.addWidget(self.pushButton_redo)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.sketchpad = Sketchpad(self.page_sketch)
        self.sketchpad.setObjectName(u"sketchpad")
        self.sketchpad.setMinimumSize(QSize(200, 200))
        self.sketchpad.setFrameShape(QFrame.Shape.NoFrame)
        self.sketchpad.setFrameShadow(QFrame.Shadow.Plain)
        self.sketchpad.setLineWidth(0)
        self.sketchpad.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sketchpad.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sketchpad.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.sketchpad.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)

        self.verticalLayout.addWidget(self.sketchpad)

        self.verticalLayout.setStretch(1, 1)
        self.stackedWidget_left.addWidget(self.page_sketch)
        self.page_image = QWidget()
        self.page_image.setObjectName(u"page_image")
        self.pushButton_back_to_drawing = QPushButton(self.page_image)
        self.pushButton_back_to_drawing.setObjectName(u"pushButton_back_to_drawing")
        self.pushButton_back_to_drawing.setGeometry(QRect(20, 20, 181, 38))
        icon3 = QIcon(QIcon.fromTheme(u"draw-freehand"))
        self.pushButton_back_to_drawing.setIcon(icon3)
        self.pushButton_back_to_drawing.setFlat(True)
        self.pushButton_back_to_drawing_2 = QPushButton(self.page_image)
        self.pushButton_back_to_drawing_2.setObjectName(u"pushButton_back_to_drawing_2")
        self.pushButton_back_to_drawing_2.setGeometry(QRect(180, 20, 181, 38))
        icon4 = QIcon(QIcon.fromTheme(u"insert-image"))
        self.pushButton_back_to_drawing_2.setIcon(icon4)
        self.pushButton_back_to_drawing_2.setFlat(True)
        self.label_10 = QLabel(self.page_image)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(20, 540, 67, 22))
        self.label_10.setText(u"<invertimage>")
        self.checkBox = QCheckBox(self.page_image)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setGeometry(QRect(120, 540, 151, 26))
        self.label_11 = QLabel(self.page_image)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(20, 580, 67, 22))
        self.label_11.setText(u"<image-rotate-right-symbolic>")
        self.label_12 = QLabel(self.page_image)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(20, 610, 67, 22))
        self.label_12.setText(u"<contrast>")
        self.label_13 = QLabel(self.page_image)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(20, 640, 67, 22))
        self.label_13.setText(u"<adjustcurves>")
        self.stackedWidget_left.addWidget(self.page_image)
        self.splitter.addWidget(self.stackedWidget_left)
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
        icon5 = QIcon(QIcon.fromTheme(u"search"))
        self.pushButton_symbol_list.setIcon(icon5)
        self.pushButton_symbol_list.setFlat(True)

        self.horizontalLayout_2.addWidget(self.pushButton_symbol_list)

        self.pushButton_menu = QPushButton(self.layoutWidget1)
        self.pushButton_menu.setObjectName(u"pushButton_menu")
        icon6 = QIcon(QIcon.fromTheme(u"application-menu"))
        self.pushButton_menu.setIcon(icon6)
        self.pushButton_menu.setFlat(True)

        self.horizontalLayout_2.addWidget(self.pushButton_menu)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.stackedWidget_right = QStackedWidget(self.layoutWidget1)
        self.stackedWidget_right.setObjectName(u"stackedWidget_right")
        self.page_classify = QWidget()
        self.page_classify.setObjectName(u"page_classify")
        self.verticalLayout_4 = QVBoxLayout(self.page_classify)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_predictions = CScrollArea(self.page_classify)
        self.scrollArea_predictions.setObjectName(u"scrollArea_predictions")
        self.scrollArea_predictions.setMinimumSize(QSize(300, 0))
        self.scrollArea_predictions.setWidgetResizable(True)
        self.widget_predictions = QWidget()
        self.widget_predictions.setObjectName(u"widget_predictions")
        self.widget_predictions.setGeometry(QRect(0, 0, 526, 630))
        self.verticalLayout_7 = QVBoxLayout(self.widget_predictions)
        self.verticalLayout_7.setSpacing(12)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_predictions.setWidget(self.widget_predictions)

        self.verticalLayout_4.addWidget(self.scrollArea_predictions)

        self.stackedWidget_right.addWidget(self.page_classify)
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

        self.verticalLayout_5.addWidget(self.widget_training_symbol, 0, Qt.AlignmentFlag.AlignHCenter)

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
        self.spinBox_max_submissions.setValue(3)

        self.horizontalLayout_6.addWidget(self.spinBox_max_submissions)


        self.verticalLayout_5.addLayout(self.horizontalLayout_6)

        self.label_2 = QLabel(self.page_train)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_5.addWidget(self.label_2, 0, Qt.AlignmentFlag.AlignHCenter)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(-1, -1, -1, 0)
        self.label_3 = QLabel(self.page_train)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_3.addWidget(self.label_3)

        self.horizontalSlider_selection_bias = QSlider(self.page_train)
        self.horizontalSlider_selection_bias.setObjectName(u"horizontalSlider_selection_bias")
        self.horizontalSlider_selection_bias.setMaximum(100)
        self.horizontalSlider_selection_bias.setValue(100)
        self.horizontalSlider_selection_bias.setOrientation(Qt.Orientation.Horizontal)
        self.horizontalSlider_selection_bias.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.horizontalSlider_selection_bias.setTickInterval(50)

        self.horizontalLayout_3.addWidget(self.horizontalSlider_selection_bias)

        self.label_4 = QLabel(self.page_train)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_3.addWidget(self.label_4)


        self.verticalLayout_5.addLayout(self.horizontalLayout_3)

        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.formLayout_3.setContentsMargins(-1, -1, -1, 24)
        self.label_7 = QLabel(self.page_train)
        self.label_7.setObjectName(u"label_7")

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.label_7)

        self.lineEdit_train_symbol = QLineEdit(self.page_train)
        self.lineEdit_train_symbol.setObjectName(u"lineEdit_train_symbol")

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.lineEdit_train_symbol)

        self.label_9 = QLabel(self.page_train)
        self.label_9.setObjectName(u"label_9")

        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.label_9)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.lineEdit_new_data_dir = QLineEdit(self.page_train)
        self.lineEdit_new_data_dir.setObjectName(u"lineEdit_new_data_dir")

        self.horizontalLayout_7.addWidget(self.lineEdit_new_data_dir)

        self.pushButton_browse_new_data_dir = QPushButton(self.page_train)
        self.pushButton_browse_new_data_dir.setObjectName(u"pushButton_browse_new_data_dir")
        icon7 = QIcon(QIcon.fromTheme(u"document-open-folder"))
        self.pushButton_browse_new_data_dir.setIcon(icon7)

        self.horizontalLayout_7.addWidget(self.pushButton_browse_new_data_dir)


        self.formLayout_3.setLayout(1, QFormLayout.FieldRole, self.horizontalLayout_7)


        self.verticalLayout_5.addLayout(self.formLayout_3)

        self.pushButton_submit = QPushButton(self.page_train)
        self.pushButton_submit.setObjectName(u"pushButton_submit")
        icon8 = QIcon(QIcon.fromTheme(u"dialog-ok"))
        self.pushButton_submit.setIcon(icon8)

        self.verticalLayout_5.addWidget(self.pushButton_submit)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(-1, 24, -1, -1)
        self.pushButton_skip = QPushButton(self.page_train)
        self.pushButton_skip.setObjectName(u"pushButton_skip")
        icon9 = QIcon(QIcon.fromTheme(u"media-skip-forward"))
        self.pushButton_skip.setIcon(icon9)

        self.horizontalLayout_4.addWidget(self.pushButton_skip)

        self.pushButton_undo_submit = QPushButton(self.page_train)
        self.pushButton_undo_submit.setObjectName(u"pushButton_undo_submit")
        self.pushButton_undo_submit.setEnabled(False)
        self.pushButton_undo_submit.setIcon(icon1)

        self.horizontalLayout_4.addWidget(self.pushButton_undo_submit)


        self.verticalLayout_5.addLayout(self.horizontalLayout_4)

        self.stackedWidget_right.addWidget(self.page_train)

        self.verticalLayout_2.addWidget(self.stackedWidget_right)

        self.splitter.addWidget(self.layoutWidget1)

        self.verticalLayout_3.addWidget(self.splitter)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.stackedWidget_left.setCurrentIndex(0)
        self.stackedWidget_right.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
#if QT_CONFIG(tooltip)
        self.pushButton_clear.setToolTip(QCoreApplication.translate("MainWindow", u"Clear (Del)", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_clear.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
#if QT_CONFIG(shortcut)
        self.pushButton_clear.setShortcut(QCoreApplication.translate("MainWindow", u"Del", None))
#endif // QT_CONFIG(shortcut)
#if QT_CONFIG(tooltip)
        self.pushButton_undo.setToolTip(QCoreApplication.translate("MainWindow", u"Undo (Ctrl + Z)", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_undo.setText(QCoreApplication.translate("MainWindow", u"Undo", None))
#if QT_CONFIG(shortcut)
        self.pushButton_undo.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Z", None))
#endif // QT_CONFIG(shortcut)
#if QT_CONFIG(tooltip)
        self.pushButton_redo.setToolTip(QCoreApplication.translate("MainWindow", u"Redo (Shift + Ctrl + Z)", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_redo.setText(QCoreApplication.translate("MainWindow", u"Redo", None))
#if QT_CONFIG(shortcut)
        self.pushButton_redo.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+Z", None))
#endif // QT_CONFIG(shortcut)
        self.pushButton_back_to_drawing.setText(QCoreApplication.translate("MainWindow", u"Back to Sketchpad", None))
        self.pushButton_back_to_drawing_2.setText(QCoreApplication.translate("MainWindow", u"New Image", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"Invert colors", None))
#if QT_CONFIG(tooltip)
        self.pushButton_symbol_list.setToolTip(QCoreApplication.translate("MainWindow", u"Symbol List (Ctrl + S)", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_symbol_list.setText(QCoreApplication.translate("MainWindow", u"Symbol List", None))
#if QT_CONFIG(shortcut)
        self.pushButton_symbol_list.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
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
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Saving to:", None))
#if QT_CONFIG(tooltip)
        self.pushButton_browse_new_data_dir.setToolTip(QCoreApplication.translate("MainWindow", u"Select directory", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_browse_new_data_dir.setText("")
#if QT_CONFIG(tooltip)
        self.pushButton_submit.setToolTip(QCoreApplication.translate("MainWindow", u"Submit (Ctrl + Space)", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_submit.setText(QCoreApplication.translate("MainWindow", u"Submit", None))
#if QT_CONFIG(shortcut)
        self.pushButton_submit.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Space", None))
#endif // QT_CONFIG(shortcut)
#if QT_CONFIG(tooltip)
        self.pushButton_skip.setToolTip(QCoreApplication.translate("MainWindow", u"Skip (Ctrl + N)", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_skip.setText(QCoreApplication.translate("MainWindow", u"Skip", None))
#if QT_CONFIG(shortcut)
        self.pushButton_skip.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+N", None))
#endif // QT_CONFIG(shortcut)
        self.pushButton_undo_submit.setText(QCoreApplication.translate("MainWindow", u"Undo Submission", None))
    # retranslateUi

