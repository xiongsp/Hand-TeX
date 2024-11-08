import platform
import sys
from functools import partial

import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtWidgets as Qw
from PySide6.QtCore import Signal
from loguru import logger

import handtex.config as cfg
import handtex.gui_utils as gu
import handtex.issue_reporter_driver as ird
import handtex.utils as ut
import handtex.state_saver as ss
from handtex import __program__, __version__
from handtex.ui_generated_files.ui_Mainwindow import Ui_MainWindow


class MainWindow(Qw.QMainWindow, Ui_MainWindow):
    config: cfg.Config = None
    debug: bool

    threadpool: Qc.QThreadPool

    hamburger_menu: Qw.QMenu
    theming_menu: Qw.QMenu

    default_palette: Qg.QPalette
    default_style: str
    theme_is_dark: ut.Shared[bool]
    theme_is_dark_changed = Signal(bool)  # When true, the new theme is dark.

    def __init__(
        self,
        debug: bool,
        train: bool,
    ) -> None:
        Qw.QMainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle(f"{__program__} {__version__}")
        self.setWindowIcon(Qg.QIcon(":/icons/logo.png"))
        self.debug = debug

        self.theme_is_dark = ut.Shared[bool](True)

        self.hamburger_menu = Qw.QMenu()

        self.initialize_ui()

        self.threadpool = Qc.QThreadPool.globalInstance()

        self.config = self.load_config()
        self.config.pretty_log()

        self.save_default_palette()
        self.load_config_theme()

        self.state_saver = ss.StateSaver("error_dialog")
        self.init_state_saver()
        self.state_saver.restore()

        Qc.QTimer.singleShot(0, self.post_init)

    def initialize_ui(self) -> None:
        if platform.system() == "Windows":
            self.setWindowIcon(gu.load_custom_icon("logo.ico"))
        else:
            self.setWindowIcon(gu.load_custom_icon("logo.svg"))

        self.set_up_hamburger_menu()

        # Make the submit button bigger.
        self.pushButton_submit.setMinimumHeight(self.pushButton_submit.height() * 1.5)

    def init_state_saver(self) -> None:
        """
        Load the state from the state saver.
        """
        self.state_saver.register(self, self.splitter)

    def post_init(self) -> None:
        """
        Post-initialization tasks, mostly stuff that needs to happen after the window is shown.
        """

        def exception_handler(exctype, value, traceback) -> None:
            gu.show_exception(
                self,
                "Uncaught Exception",
                "An uncaught exception was raised.",
                error_bundle=(exctype, value, traceback),
            )

        sys.excepthook = exception_handler

    def closeEvent(self, event: Qg.QCloseEvent) -> None:
        """
        Notify config on close.
        """
        logger.info("Closing window.")
        self.state_saver.save()
        if self.threadpool.activeThreadCount():
            # Process Qt events so that the message shows up.
            Qc.QCoreApplication.processEvents()
            self.threadpool.waitForDone()

        event.accept()

    def load_config(self) -> cfg.Config:
        """
        Load the config if there is one, handling errors as needed.

        :return: The loaded or default config.
        """
        # Check if there is a config at all.
        config_path = ut.get_config_path()
        if not config_path.exists():
            logger.info(f"No config found at {config_path}.")
            return cfg.Config()

        config, recoverable_exceptions, errors, critical_errors = cfg.load_config(config_path)

        # Critical errors force handtex to nuke the config and use the default.
        if critical_errors:
            backup_path = ut.backup_file(config_path)
            # Format them like: "ValueError: 'lang_from' must be a string."
            critical_errors_str = "\n\n".join(map(str, critical_errors))
            response = gu.show_critical(
                self,
                "Critical Configuration Error",
                f"Failed to load the config file.\n\n"
                f"A backup of the config file was created at \n{backup_path}.\n\n"
                f"Proceed with the default configuration?",
                detailedText=f"Critical errors:\n\n{critical_errors_str}",
            )
            if response == Qw.QMessageBox.Abort:
                logger.critical("User aborted due to critical config errors.")
                Qw.QApplication.instance().quit()  # Embrace death.
                raise SystemExit(255)
            return config

        # Errors and recoverable exceptions can occur at the same time, they don't
        # require throwing out the whole config.
        if errors:
            backup_path = ut.backup_file(config_path)
            errors_str = "\n\n".join(map(str, errors))
            response = gu.show_critical(
                self,
                "Configuration Error",
                f"Failed to load parts of the config file.\n\n"
                f"A backup of the config file was created at \n{backup_path}.\n\n"
                f"Proceed with the default configuration for affected sections?",
                detailedText=f"Critical errors:\n\n{errors_str}",
            )
            if response == Qw.QMessageBox.Abort:
                logger.critical("User aborted due to critical config errors.")
                Qw.QApplication.instance().quit()  # Embrace death.
                raise SystemExit(255)

        if recoverable_exceptions:
            recoverable_exceptions_str = "\n\n".join(map(str, recoverable_exceptions))
            gu.show_info(
                self,
                "Config Warnings",
                f"Minor issues were found and corrected in the config file.",
                detailedText=f"Recoverable exceptions:\n\n{recoverable_exceptions_str}",
            )

        return config

    def set_up_hamburger_menu(self) -> None:
        """
        This is the hamburger menu on the main window.
        It contains several menu-bar-esque actions.
        """
        self.pushButton_menu.setMenu(self.hamburger_menu)
        # Add theming menu.
        self.theming_menu = self.hamburger_menu.addMenu(
            Qg.QIcon.fromTheme("games-config-theme"), "Theme"
        )
        themes = [("", "System")]
        themes.extend(ut.get_available_themes())
        for theme, name in themes:
            action = Qg.QAction(name, self)
            action.setCheckable(True)
            action.theme = theme
            action.triggered.connect(partial(self.set_theme, theme))
            self.theming_menu.addAction(action)

        if self.debug:
            # Add an intentional crash button.
            self.hamburger_menu.addSeparator()
            action = Qg.QAction(Qg.QIcon.fromTheme("tools-report-bug"), "Simulate crash", self)
            action.triggered.connect(self.simulate_crash)
            logger.warning("Debug mode active.")
            self.hamburger_menu.addAction(action)

    def open_log_viewer(self) -> None:
        logger.debug("Opening issue reporter.")
        issue_reporter = ird.IssueReporter(self)
        issue_reporter.exec()

    # =========================================== Theming ===========================================

    def save_default_palette(self) -> None:
        self.default_palette = self.palette()
        # Patch palette to use the text color with 50% opacity for placeholder text.
        placeholder_color = self.default_palette.color(Qg.QPalette.Inactive, Qg.QPalette.Text)
        placeholder_color.setAlphaF(0.5)
        logger.debug(f"Placeholder color: {placeholder_color.name()}")
        self.default_palette.setColor(Qg.QPalette.PlaceholderText, placeholder_color)
        # self.default_icon_theme = Qg.QIcon.themeName()
        self.default_style = Qw.QApplication.style().objectName()

    def load_config_theme(self) -> None:
        """
        Load the theme specified in the config, or the system theme if none.
        """
        theme = self.config.gui_theme
        self.set_theme(theme)

    def set_theme(self, theme: str = "") -> None:
        """
        Apply the given theme to the application, or if none, revert to the default theme.
        """
        palette: Qg.QPalette

        if not theme:
            logger.info(f"Using system theme.")
            palette = self.default_palette
            # Qg.QIcon.setThemeName(self.default_icon_theme)
            # Check if we need to restore the style.
            if Qw.QApplication.style().objectName() != self.default_style:
                Qw.QApplication.setStyle(self.default_style)
        else:
            logger.info(f"Using theme: {theme}")
            palette = gu.load_color_palette(theme)

            Qg.QIcon.setThemeName(theme)
            if platform.system() == "Windows":
                Qw.QApplication.setStyle("Fusion")

        self.setPalette(palette)
        Qw.QApplication.setPalette(self.palette())

        # Check the brightness of the background color to determine if the theme is dark.
        # This is a heuristic, but it works well enough.
        background_color = palette.color(Qg.QPalette.Window)
        self.theme_is_dark.set(background_color.lightness() < 128)
        logger.info(f"Theme is dark: {self.theme_is_dark.get()}")
        self.theme_is_dark_changed.emit(self.theme_is_dark)

        # Also just setting the icon theme here, since with qt6 even breeze dark is having issues.
        # Update the fallback icon theme accordingly.
        if self.theme_is_dark.get():
            Qg.QIcon.setFallbackThemeName("breeze-dark")
            Qg.QIcon.setThemeName("breeze-dark")
            logger.info("Setting icon theme to breeze-dark.")
        else:
            Qg.QIcon.setFallbackThemeName("breeze")
            Qg.QIcon.setThemeName("breeze")
            logger.info("Setting icon theme to breeze.")

        # Toggle the theme menu items.
        for action in self.theming_menu.actions():
            action.setChecked(action.theme == theme)

        self.update()
        event = Qc.QEvent(Qc.QEvent.PaletteChange)

        # Update the config it necessary.
        prev_value = self.config.gui_theme
        if prev_value != theme:
            self.config.gui_theme = theme
            self.config.save()

    def changeEvent(self, event) -> None:
        """
        Listen for palette change events to notify all widgets.
        """
        if event.type() == Qc.QEvent.ApplicationPaletteChange:
            background_color = self.palette().color(Qg.QPalette.Window)
            self.theme_is_dark.set(background_color.lightness() < 128)
            logger.info(f"Theme is dark: {self.theme_is_dark.get()}")
            self.theme_is_dark_changed.emit(self.theme_is_dark)

    def simulate_crash(self) -> None:
        """
        Simulate a crash by raising an exception.
        """
        raise Exception("This is a simulated crash.")
