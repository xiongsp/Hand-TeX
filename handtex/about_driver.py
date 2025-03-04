import datetime

import PySide6
import PySide6.QtGui as Qg
import PySide6.QtWidgets as Qw
from PySide6.QtCore import Qt
import PySide6.QtCore as Qc
import PySide6.QtSvgWidgets as Qsw

import handtex.license_driver as ld
import handtex.gui_utils as gu
from handtex import __version__
from handtex.ui_generated_files.ui_About import Ui_About


class AboutWidget(Qw.QWidget, Ui_About):
    """
    Displays the about information.
    """

    def __init__(
        self,
        parent: Qw.QWidget | None = None,
    ):
        """
        Initialize the widget.

        :param parent: The parent widget.
        """
        Qw.QWidget.__init__(self, parent)
        self.setupUi(self)
        self.setWindowFlag(Qt.Window)
        self.setWindowIcon(gu.load_custom_icon("logo"))

        self.label_license.linkActivated.connect(self.open_license)

        copyright_str = f"© 2025"
        if (until := datetime.datetime.now().year) > 2025:
            copyright_str += f"–{until}"

        self.label_copyright.setText(copyright_str)

        self.label_version.setText(__version__)
        self.label_toolkit.setText(f"PySide (Qt) {PySide6.__version__}")

        self.label_logo.setPixmap(gu.load_custom_icon("logo").pixmap(200, 200))

    def open_license(self) -> None:
        """
        Open the license dialog.
        """
        license_widget = ld.LicenseDialog(self)
        license_widget.show()
