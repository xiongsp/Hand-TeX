# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SymbolList.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QListView,
    QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
    QSpacerItem, QSpinBox, QSplitter, QVBoxLayout,
    QWidget)

from handtex.CustomQ.CRegexLineEdit import RegexLineEdit

class Ui_SymbolList(object):
    def setupUi(self, SymbolList):
        if not SymbolList.objectName():
            SymbolList.setObjectName(u"SymbolList")
        SymbolList.resize(1300, 600)
        self.verticalLayout_3 = QVBoxLayout(SymbolList)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, -1, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(6, -1, 6, -1)
        self.comboBox_search_field = QComboBox(SymbolList)
        self.comboBox_search_field.addItem("")
        self.comboBox_search_field.addItem("")
        self.comboBox_search_field.setObjectName(u"comboBox_search_field")

        self.horizontalLayout.addWidget(self.comboBox_search_field)

        self.lineEdit_search = RegexLineEdit(SymbolList)
        self.lineEdit_search.setObjectName(u"lineEdit_search")
        self.lineEdit_search.setClearButtonEnabled(True)

        self.horizontalLayout.addWidget(self.lineEdit_search)

        self.pushButton_filter = QPushButton(SymbolList)
        self.pushButton_filter.setObjectName(u"pushButton_filter")
        self.pushButton_filter.setText(u"Filters")
        icon = QIcon(QIcon.fromTheme(u"dialog-filters"))
        self.pushButton_filter.setIcon(icon)
        self.pushButton_filter.setCheckable(True)

        self.horizontalLayout.addWidget(self.pushButton_filter)

        self.pushButton_clear_filters = QPushButton(SymbolList)
        self.pushButton_clear_filters.setObjectName(u"pushButton_clear_filters")
        icon1 = QIcon(QIcon.fromTheme(u"edit-clear"))
        self.pushButton_clear_filters.setIcon(icon1)

        self.horizontalLayout.addWidget(self.pushButton_clear_filters)

        self.comboBox_sort = QComboBox(SymbolList)
        self.comboBox_sort.addItem("")
        self.comboBox_sort.addItem("")
        self.comboBox_sort.addItem("")
        self.comboBox_sort.setObjectName(u"comboBox_sort")

        self.horizontalLayout.addWidget(self.comboBox_sort)

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
        self.horizontalLayout.setStretch(5, 1)

        self.verticalLayout_3.addLayout(self.horizontalLayout)

        self.widget_filters = QWidget(SymbolList)
        self.widget_filters.setObjectName(u"widget_filters")
        self.horizontalLayout_2 = QHBoxLayout(self.widget_filters)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(6, 12, 6, 12)
        self.comboBox_case = QComboBox(self.widget_filters)
        self.comboBox_case.addItem("")
        self.comboBox_case.addItem("")
        self.comboBox_case.addItem("")
        self.comboBox_case.setObjectName(u"comboBox_case")

        self.horizontalLayout_2.addWidget(self.comboBox_case)

        self.comboBox_mode = QComboBox(self.widget_filters)
        self.comboBox_mode.addItem("")
        self.comboBox_mode.addItem("")
        self.comboBox_mode.addItem("")
        self.comboBox_mode.setObjectName(u"comboBox_mode")

        self.horizontalLayout_2.addWidget(self.comboBox_mode)

        self.pushButton_packages = QPushButton(self.widget_filters)
        self.pushButton_packages.setObjectName(u"pushButton_packages")

        self.horizontalLayout_2.addWidget(self.pushButton_packages)

        self.pushButton_encodings = QPushButton(self.widget_filters)
        self.pushButton_encodings.setObjectName(u"pushButton_encodings")

        self.horizontalLayout_2.addWidget(self.pushButton_encodings)

        self.comboBox_symmetry = QComboBox(self.widget_filters)
        self.comboBox_symmetry.addItem("")
        self.comboBox_symmetry.addItem("")
        self.comboBox_symmetry.addItem("")
        self.comboBox_symmetry.setObjectName(u"comboBox_symmetry")

        self.horizontalLayout_2.addWidget(self.comboBox_symmetry)

        self.line = QFrame(self.widget_filters)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.VLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.horizontalLayout_2.addWidget(self.line)

        self.comboBox_grouping = QComboBox(self.widget_filters)
        self.comboBox_grouping.addItem("")
        self.comboBox_grouping.addItem("")
        self.comboBox_grouping.addItem("")
        self.comboBox_grouping.addItem("")
        self.comboBox_grouping.addItem("")
        self.comboBox_grouping.addItem("")
        self.comboBox_grouping.setObjectName(u"comboBox_grouping")

        self.horizontalLayout_2.addWidget(self.comboBox_grouping)

        self.label_4 = QLabel(self.widget_filters)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_2.addWidget(self.label_4)

        self.spinBox_group_min_size = QSpinBox(self.widget_filters)
        self.spinBox_group_min_size.setObjectName(u"spinBox_group_min_size")
        self.spinBox_group_min_size.setMinimum(1)
        self.spinBox_group_min_size.setMaximum(9999)

        self.horizontalLayout_2.addWidget(self.spinBox_group_min_size)

        self.label_9 = QLabel(self.widget_filters)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_2.addWidget(self.label_9)

        self.spinBox_group_max_size = QSpinBox(self.widget_filters)
        self.spinBox_group_max_size.setObjectName(u"spinBox_group_max_size")
        self.spinBox_group_max_size.setMinimum(1)
        self.spinBox_group_max_size.setMaximum(9999)
        self.spinBox_group_max_size.setValue(9999)

        self.horizontalLayout_2.addWidget(self.spinBox_group_max_size)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.verticalLayout_3.addWidget(self.widget_filters)

        self.splitter = QSplitter(SymbolList)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.listWidget = QListWidget(self.splitter)
        self.listWidget.setObjectName(u"listWidget")
        self.listWidget.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.listWidget.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.listWidget.setMovement(QListView.Movement.Static)
        self.listWidget.setResizeMode(QListView.ResizeMode.Adjust)
        self.listWidget.setLayoutMode(QListView.LayoutMode.Batched)
        self.listWidget.setViewMode(QListView.ViewMode.IconMode)
        self.listWidget.setUniformItemSizes(True)
        self.splitter.addWidget(self.listWidget)
        self.verticalLayoutWidget = QWidget(self.splitter)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSpacing(24)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(6, 0, 0, 0)
        self.label_command = QLabel(self.verticalLayoutWidget)
        self.label_command.setObjectName(u"label_command")
        self.label_command.setText(u"<command>")
        self.label_command.setWordWrap(True)
        self.label_command.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse|Qt.TextInteractionFlag.TextSelectableByKeyboard|Qt.TextInteractionFlag.TextSelectableByMouse)

        self.verticalLayout.addWidget(self.label_command, 0, Qt.AlignmentFlag.AlignHCenter)

        self.widget_symbol_view = QSvgWidget(self.verticalLayoutWidget)
        self.widget_symbol_view.setObjectName(u"widget_symbol_view")
        self.widget_symbol_view.setMinimumSize(QSize(64, 64))
        self.verticalLayout_2 = QVBoxLayout(self.widget_symbol_view)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")

        self.verticalLayout.addWidget(self.widget_symbol_view, 0, Qt.AlignmentFlag.AlignHCenter)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label = QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)

        self.label_package_label = QLabel(self.verticalLayoutWidget)
        self.label_package_label.setObjectName(u"label_package_label")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_package_label)

        self.label_package = QLabel(self.verticalLayoutWidget)
        self.label_package.setObjectName(u"label_package")
        self.label_package.setText(u"<package>")
        self.label_package.setWordWrap(True)
        self.label_package.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse|Qt.TextInteractionFlag.TextSelectableByKeyboard|Qt.TextInteractionFlag.TextSelectableByMouse)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.label_package)

        self.label_3 = QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_3)

        self.label_mode = QLabel(self.verticalLayoutWidget)
        self.label_mode.setObjectName(u"label_mode")
        self.label_mode.setText(u"<mode>")
        self.label_mode.setWordWrap(True)
        self.label_mode.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse|Qt.TextInteractionFlag.TextSelectableByKeyboard|Qt.TextInteractionFlag.TextSelectableByMouse)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.label_mode)

        self.label_fontenc_label = QLabel(self.verticalLayoutWidget)
        self.label_fontenc_label.setObjectName(u"label_fontenc_label")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.label_fontenc_label)

        self.label_fontenc = QLabel(self.verticalLayoutWidget)
        self.label_fontenc.setObjectName(u"label_fontenc")
        self.label_fontenc.setText(u"<fontenc>")
        self.label_fontenc.setWordWrap(True)
        self.label_fontenc.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse|Qt.TextInteractionFlag.TextSelectableByKeyboard|Qt.TextInteractionFlag.TextSelectableByMouse)

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.label_fontenc)

        self.label_7 = QLabel(self.verticalLayoutWidget)
        self.label_7.setObjectName(u"label_7")

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.label_7)

        self.label_similar = QLabel(self.verticalLayoutWidget)
        self.label_similar.setObjectName(u"label_similar")
        self.label_similar.setText(u"<similar>")
        self.label_similar.setWordWrap(True)
        self.label_similar.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse|Qt.TextInteractionFlag.TextSelectableByKeyboard|Qt.TextInteractionFlag.TextSelectableByMouse)

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.label_similar)

        self.label_2 = QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(6, QFormLayout.LabelRole, self.label_2)

        self.label_self_symmetry = QLabel(self.verticalLayoutWidget)
        self.label_self_symmetry.setObjectName(u"label_self_symmetry")
        self.label_self_symmetry.setWordWrap(True)

        self.formLayout.setWidget(6, QFormLayout.FieldRole, self.label_self_symmetry)

        self.label_5 = QLabel(self.verticalLayoutWidget)
        self.label_5.setObjectName(u"label_5")

        self.formLayout.setWidget(7, QFormLayout.LabelRole, self.label_5)

        self.label_other_symmetry = QLabel(self.verticalLayoutWidget)
        self.label_other_symmetry.setObjectName(u"label_other_symmetry")
        self.label_other_symmetry.setWordWrap(True)

        self.formLayout.setWidget(7, QFormLayout.FieldRole, self.label_other_symmetry)

        self.label_id = QLabel(self.verticalLayoutWidget)
        self.label_id.setObjectName(u"label_id")
        self.label_id.setText(u"<symbol id>")
        self.label_id.setWordWrap(True)
        self.label_id.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse|Qt.TextInteractionFlag.TextSelectableByKeyboard|Qt.TextInteractionFlag.TextSelectableByMouse)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.label_id)

        self.label_xelatex_required_label = QLabel(self.verticalLayoutWidget)
        self.label_xelatex_required_label.setObjectName(u"label_xelatex_required_label")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_xelatex_required_label)

        self.label_xelatex_required = QLabel(self.verticalLayoutWidget)
        self.label_xelatex_required.setObjectName(u"label_xelatex_required")
        self.label_xelatex_required.setWordWrap(True)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.label_xelatex_required)

        self.label_6 = QLabel(self.verticalLayoutWidget)
        self.label_6.setObjectName(u"label_6")

        self.formLayout.setWidget(8, QFormLayout.LabelRole, self.label_6)

        self.label_negation = QLabel(self.verticalLayoutWidget)
        self.label_negation.setObjectName(u"label_negation")
        self.label_negation.setText(u"<symbol>")
        self.label_negation.setWordWrap(True)

        self.formLayout.setWidget(8, QFormLayout.FieldRole, self.label_negation)

        self.label_8 = QLabel(self.verticalLayoutWidget)
        self.label_8.setObjectName(u"label_8")

        self.formLayout.setWidget(9, QFormLayout.LabelRole, self.label_8)

        self.label_inside_shape = QLabel(self.verticalLayoutWidget)
        self.label_inside_shape.setObjectName(u"label_inside_shape")
        self.label_inside_shape.setText(u"<shape>")
        self.label_inside_shape.setWordWrap(True)

        self.formLayout.setWidget(9, QFormLayout.FieldRole, self.label_inside_shape)


        self.verticalLayout.addLayout(self.formLayout)

        self.verticalLayout.setStretch(2, 1)
        self.splitter.addWidget(self.verticalLayoutWidget)

        self.verticalLayout_3.addWidget(self.splitter)

        self.verticalLayout_3.setStretch(2, 1)
        QWidget.setTabOrder(self.comboBox_search_field, self.lineEdit_search)
        QWidget.setTabOrder(self.lineEdit_search, self.pushButton_filter)
        QWidget.setTabOrder(self.pushButton_filter, self.pushButton_clear_filters)
        QWidget.setTabOrder(self.pushButton_clear_filters, self.comboBox_sort)
        QWidget.setTabOrder(self.comboBox_sort, self.comboBox_case)
        QWidget.setTabOrder(self.comboBox_case, self.comboBox_mode)
        QWidget.setTabOrder(self.comboBox_mode, self.pushButton_packages)
        QWidget.setTabOrder(self.pushButton_packages, self.pushButton_encodings)
        QWidget.setTabOrder(self.pushButton_encodings, self.comboBox_symmetry)
        QWidget.setTabOrder(self.comboBox_symmetry, self.comboBox_grouping)
        QWidget.setTabOrder(self.comboBox_grouping, self.spinBox_group_min_size)
        QWidget.setTabOrder(self.spinBox_group_min_size, self.spinBox_group_max_size)
        QWidget.setTabOrder(self.spinBox_group_max_size, self.listWidget)

        self.retranslateUi(SymbolList)

        QMetaObject.connectSlotsByName(SymbolList)
    # setupUi

    def retranslateUi(self, SymbolList):
        SymbolList.setWindowTitle(QCoreApplication.translate("SymbolList", u"Symbol List", None))
        self.comboBox_search_field.setItemText(0, QCoreApplication.translate("SymbolList", u"Command", None))
        self.comboBox_search_field.setItemText(1, QCoreApplication.translate("SymbolList", u"Symbol ID", None))

        self.lineEdit_search.setPlaceholderText(QCoreApplication.translate("SymbolList", u"Search...", None))
        self.pushButton_clear_filters.setText(QCoreApplication.translate("SymbolList", u"Clear Filters", None))
        self.comboBox_sort.setItemText(0, QCoreApplication.translate("SymbolList", u"Sort by Groups", None))
        self.comboBox_sort.setItemText(1, QCoreApplication.translate("SymbolList", u"Sorted Ascending", None))
        self.comboBox_sort.setItemText(2, QCoreApplication.translate("SymbolList", u"Sorted Descending", None))

        self.label_11.setText(QCoreApplication.translate("SymbolList", u"Found:", None))
        self.comboBox_case.setItemText(0, QCoreApplication.translate("SymbolList", u"Case Insensitive", None))
        self.comboBox_case.setItemText(1, QCoreApplication.translate("SymbolList", u"Case Sensitive", None))
        self.comboBox_case.setItemText(2, QCoreApplication.translate("SymbolList", u"Regex", None))

        self.comboBox_mode.setItemText(0, QCoreApplication.translate("SymbolList", u"Math & Textmode", None))
        self.comboBox_mode.setItemText(1, QCoreApplication.translate("SymbolList", u"Mathmode", None))
        self.comboBox_mode.setItemText(2, QCoreApplication.translate("SymbolList", u"Textmode", None))

        self.pushButton_packages.setText(QCoreApplication.translate("SymbolList", u"Packages", None))
        self.pushButton_encodings.setText(QCoreApplication.translate("SymbolList", u"Font Encodings", None))
        self.comboBox_symmetry.setItemText(0, QCoreApplication.translate("SymbolList", u"Ignore Symmetry", None))
        self.comboBox_symmetry.setItemText(1, QCoreApplication.translate("SymbolList", u"Symmetric Symbols", None))
        self.comboBox_symmetry.setItemText(2, QCoreApplication.translate("SymbolList", u"Asymmetric Symbols", None))

        self.comboBox_grouping.setItemText(0, QCoreApplication.translate("SymbolList", u"No Grouping", None))
        self.comboBox_grouping.setItemText(1, QCoreApplication.translate("SymbolList", u"Group by Package", None))
        self.comboBox_grouping.setItemText(2, QCoreApplication.translate("SymbolList", u"Group by Similarity", None))
        self.comboBox_grouping.setItemText(3, QCoreApplication.translate("SymbolList", u"Group by Symmetry", None))
        self.comboBox_grouping.setItemText(4, QCoreApplication.translate("SymbolList", u"Group by Negation", None))
        self.comboBox_grouping.setItemText(5, QCoreApplication.translate("SymbolList", u"Group by Inside", None))

        self.label_4.setText(QCoreApplication.translate("SymbolList", u"Minimimum Group Size:", None))
        self.label_9.setText(QCoreApplication.translate("SymbolList", u"Maximum Group Size:", None))
        self.label.setText(QCoreApplication.translate("SymbolList", u"Symbol ID:", None))
        self.label_package_label.setText(QCoreApplication.translate("SymbolList", u"Package:", None))
        self.label_3.setText(QCoreApplication.translate("SymbolList", u"Mode:", None))
        self.label_fontenc_label.setText(QCoreApplication.translate("SymbolList", u"Font Encoding:", None))
        self.label_7.setText(QCoreApplication.translate("SymbolList", u"Similar to:", None))
        self.label_2.setText(QCoreApplication.translate("SymbolList", u"Self-symmetry:", None))
        self.label_self_symmetry.setText(QCoreApplication.translate("SymbolList", u"<yes/no>", None))
        self.label_5.setText(QCoreApplication.translate("SymbolList", u"Symmetrical to:", None))
        self.label_other_symmetry.setText(QCoreApplication.translate("SymbolList", u"<others>", None))
        self.label_xelatex_required_label.setText(QCoreApplication.translate("SymbolList", u"Compiler:", None))
        self.label_xelatex_required.setText(QCoreApplication.translate("SymbolList", u"XeLaTeX or LuaLaTeX required", None))
        self.label_6.setText(QCoreApplication.translate("SymbolList", u"Negation of:", None))
        self.label_8.setText(QCoreApplication.translate("SymbolList", u"Inside of:", None))
    # retranslateUi

