# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SymbolList.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListView,
    QListWidget, QListWidgetItem, QSizePolicy, QSpacerItem,
    QSplitter, QVBoxLayout, QWidget)

class Ui_SymbolList(object):
    def setupUi(self, SymbolList):
        if not SymbolList.objectName():
            SymbolList.setObjectName(u"SymbolList")
        SymbolList.resize(900, 600)
        self.verticalLayout_3 = QVBoxLayout(SymbolList)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, -1, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(6, -1, 6, -1)
        self.comboBox_search_mode = QComboBox(SymbolList)
        self.comboBox_search_mode.addItem("")
        self.comboBox_search_mode.addItem("")
        self.comboBox_search_mode.addItem("")
        self.comboBox_search_mode.setObjectName(u"comboBox_search_mode")

        self.horizontalLayout.addWidget(self.comboBox_search_mode)

        self.lineEdit_search = QLineEdit(SymbolList)
        self.lineEdit_search.setObjectName(u"lineEdit_search")

        self.horizontalLayout.addWidget(self.lineEdit_search)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.label_11 = QLabel(SymbolList)
        self.label_11.setObjectName(u"label_11")

        self.horizontalLayout.addWidget(self.label_11)

        self.label_count = QLabel(SymbolList)
        self.label_count.setObjectName(u"label_count")
        self.label_count.setText(u"<count>")

        self.horizontalLayout.addWidget(self.label_count)

        self.horizontalLayout.setStretch(1, 2)
        self.horizontalLayout.setStretch(2, 1)

        self.verticalLayout_3.addLayout(self.horizontalLayout)

        self.splitter = QSplitter(SymbolList)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.listWidget = QListWidget(self.splitter)
        self.listWidget.setObjectName(u"listWidget")
        self.listWidget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.listWidget.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.listWidget.setMovement(QListView.Static)
        self.listWidget.setResizeMode(QListView.Adjust)
        self.listWidget.setLayoutMode(QListView.Batched)
        self.listWidget.setViewMode(QListView.IconMode)
        self.listWidget.setUniformItemSizes(True)
        self.splitter.addWidget(self.listWidget)
        self.verticalLayoutWidget = QWidget(self.splitter)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSpacing(24)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(6, 0, 0, 0)
        self.label_id = QLabel(self.verticalLayoutWidget)
        self.label_id.setObjectName(u"label_id")
        self.label_id.setText(u"<symbol id>")
        self.label_id.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.verticalLayout.addWidget(self.label_id)

        self.widget_symbol_view = QSvgWidget(self.verticalLayoutWidget)
        self.widget_symbol_view.setObjectName(u"widget_symbol_view")
        self.widget_symbol_view.setMinimumSize(QSize(64, 64))
        self.verticalLayout_2 = QVBoxLayout(self.widget_symbol_view)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")

        self.verticalLayout.addWidget(self.widget_symbol_view, 0, Qt.AlignHCenter)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label = QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)

        self.label_command = QLabel(self.verticalLayoutWidget)
        self.label_command.setObjectName(u"label_command")
        self.label_command.setText(u"<command>")
        self.label_command.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.label_command)

        self.label_package_label = QLabel(self.verticalLayoutWidget)
        self.label_package_label.setObjectName(u"label_package_label")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_package_label)

        self.label_package = QLabel(self.verticalLayoutWidget)
        self.label_package.setObjectName(u"label_package")
        self.label_package.setText(u"<package>")
        self.label_package.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.label_package)

        self.label_3 = QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_3)

        self.label_mode = QLabel(self.verticalLayoutWidget)
        self.label_mode.setObjectName(u"label_mode")
        self.label_mode.setText(u"<mode>")
        self.label_mode.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.label_mode)

        self.label_fontenc_label = QLabel(self.verticalLayoutWidget)
        self.label_fontenc_label.setObjectName(u"label_fontenc_label")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_fontenc_label)

        self.label_fontenc = QLabel(self.verticalLayoutWidget)
        self.label_fontenc.setObjectName(u"label_fontenc")
        self.label_fontenc.setText(u"<fontenc>")
        self.label_fontenc.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.label_fontenc)

        self.label_7 = QLabel(self.verticalLayoutWidget)
        self.label_7.setObjectName(u"label_7")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.label_7)

        self.label_similar = QLabel(self.verticalLayoutWidget)
        self.label_similar.setObjectName(u"label_similar")
        self.label_similar.setText(u"<similar>")
        self.label_similar.setWordWrap(True)
        self.label_similar.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.label_similar)


        self.verticalLayout.addLayout(self.formLayout)

        self.verticalLayout.setStretch(2, 1)
        self.splitter.addWidget(self.verticalLayoutWidget)

        self.verticalLayout_3.addWidget(self.splitter)

        self.verticalLayout_3.setStretch(1, 1)

        self.retranslateUi(SymbolList)

        QMetaObject.connectSlotsByName(SymbolList)
    # setupUi

    def retranslateUi(self, SymbolList):
        SymbolList.setWindowTitle(QCoreApplication.translate("SymbolList", u"Symbol List", None))
        self.comboBox_search_mode.setItemText(0, QCoreApplication.translate("SymbolList", u"Command", None))
        self.comboBox_search_mode.setItemText(1, QCoreApplication.translate("SymbolList", u"Symbol ID", None))
        self.comboBox_search_mode.setItemText(2, QCoreApplication.translate("SymbolList", u"Similar to", None))

        self.lineEdit_search.setPlaceholderText(QCoreApplication.translate("SymbolList", u"Search...", None))
        self.label_11.setText(QCoreApplication.translate("SymbolList", u"Found:", None))
        self.label.setText(QCoreApplication.translate("SymbolList", u"Command:", None))
        self.label_package_label.setText(QCoreApplication.translate("SymbolList", u"Package:", None))
        self.label_3.setText(QCoreApplication.translate("SymbolList", u"Mode:", None))
        self.label_fontenc_label.setText(QCoreApplication.translate("SymbolList", u"Font Encoding:", None))
        self.label_7.setText(QCoreApplication.translate("SymbolList", u"Similar to:", None))
    # retranslateUi

