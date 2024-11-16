import platform
import sys
from functools import partial
import time

import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtWidgets as Qw
import PySide6.QtSvgWidgets as Qsw
from PySide6.QtCore import Signal
from loguru import logger

import handtex.config as cfg
import handtex.gui_utils as gu
import handtex.structures as st
import handtex.issue_reporter_driver as ird
import handtex.utils as ut
import handtex.state_saver as ss
import handtex.data_recorder as dr
import handtex.symbol_list as sl
from handtex import __program__, __version__
from handtex.ui_generated_files.ui_Mainwindow import Ui_MainWindow

import training.image_gen as ig
import training.train as trn
import training.inference as inf


class MainWindow(Qw.QMainWindow, Ui_MainWindow):
    config: cfg.Config = None
    debug: bool

    symbols: dict[str, st.Symbol]

    threadpool: Qc.QThreadPool

    hamburger_menu: Qw.QMenu
    theming_menu: Qw.QMenu
    stroke_width_menu: Qw.QMenu

    default_palette: Qg.QPalette
    default_style: str
    theme_is_dark: ut.Shared[bool]
    theme_is_dark_changed = Signal(bool)  # When true, the new theme is dark.

    state_saver: ss.StateSaver

    # Detection:
    model: trn.CNN
    label_decoder: dict[int, str]
    current_predictions: list[tuple[str, float]]

    # Training:
    train: bool
    data_recorder: dr.DataRecorder
    current_symbol: st.Symbol | None
    submission_count: int
    has_submission = Signal(bool)

    # Symbol list:
    symbol_list: sl.SymbolList | None

    # Lookalikes:
    similar_symbols: dict[str, tuple[str, ...]]

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

        self.current_predictions = []

        self.train = train
        self.submission_count = 1
        self.current_symbol = None

        self.symbol_list = None

        self.config = self.load_config()
        self.config.pretty_log()

        self.theme_is_dark = ut.Shared[bool](True)

        self.hamburger_menu = Qw.QMenu()

        self.initialize_ui()

        self.threadpool = Qc.QThreadPool.globalInstance()

        self.save_default_palette()
        self.load_config_theme()

        self.symbols = ut.load_symbols()
        self.similar_symbols = ut.load_symbol_metadata_similarity()

        if self.train:
            logger.info("Training mode active.")
            self.stackedWidget.setCurrentIndex(1)
            self.data_recorder = dr.DataRecorder(self.symbols, self.has_submission)
            self.data_recorder.has_submissions.connect(self.pushButton_undo_submit.setEnabled)
        else:
            self.sketchpad.new_drawing.connect(self.detect_symbol)
            self.theme_is_dark_changed.connect(self.show_predictions)
            self.model, self.label_decoder = inf.load_model_and_decoder(
                "../training/cnn_model.pt", trn.num_classes, "../training/encodings.txt"
            )

        self.state_saver = ss.StateSaver("mainwindow")
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

        # Make training symbol name bigger.
        font = self.label_training_name.font()
        font.setPointSize(font.pointSize() * 2)
        self.label_training_name.setFont(font)

        # Connect slots.
        self.pushButton_clear.clicked.connect(self.sketchpad.clear)
        self.pushButton_undo.clicked.connect(self.sketchpad.undo)
        self.pushButton_redo.clicked.connect(self.sketchpad.redo)
        self.sketchpad.can_undo.connect(self.pushButton_undo.setEnabled)
        self.sketchpad.can_undo.connect(self.pushButton_clear.setEnabled)
        self.sketchpad.can_redo.connect(self.pushButton_redo.setEnabled)
        self.spinBox_max_submissions.valueChanged.connect(self.update_submission_count)

        self.pushButton_submit.clicked.connect(self.submit_training_drawing)
        self.pushButton_skip.clicked.connect(self.get_next_symbol)
        self.pushButton_undo_submit.clicked.connect(self.previous_symbol)

        self.pushButton_symbol_list.clicked.connect(self.open_symbol_list)

    def init_state_saver(self) -> None:
        """
        Load the state from the state saver.
        """
        self.state_saver.register(
            self, self.splitter, self.horizontalSlider_selection_bias, self.spinBox_max_submissions
        )

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

        self.sketchpad.set_pen_width(self.config.stroke_width)

        if self.train:
            self.get_next_symbol()

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
        logger.debug("Loading config.")
        # Check if there is a config at all.
        config_path = ut.get_config_path()
        if not config_path.exists():
            logger.info(f"No config found at {config_path}.")
            return cfg.Config()

        config, recoverable_exceptions, critical_errors = cfg.load_config(config_path)

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

        # Recoverable exceptions can occur at the same time, they don't
        # require throwing out the whole config.
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
        theme_action_group = Qg.QActionGroup(self)
        theme_action_group.setExclusive(True)
        for theme, name in themes:
            action = Qg.QAction(name, self)
            action.setCheckable(True)
            theme_action_group.addAction(action)
            action.theme = theme
            action.triggered.connect(partial(self.set_theme, theme))
            self.theming_menu.addAction(action)

        # Add stroke width selection.
        self.stroke_width_menu = self.hamburger_menu.addMenu(
            Qg.QIcon.fromTheme("edit-line-width"), "Line Thickness"
        )
        supported_stroke_widths = [2, 4, 6, 8, 12, 16, 20]
        stroke_action_group = Qg.QActionGroup(self)
        stroke_action_group.setExclusive(True)
        for width in supported_stroke_widths:
            action = Qg.QAction(str(width), self)
            action.setCheckable(True)
            stroke_action_group.addAction(action)
            action.width = width
            action.triggered.connect(partial(self.change_pen_width, width))
            if width == self.config.stroke_width:
                action.setChecked(True)
            self.stroke_width_menu.addAction(action)

        # Offer opening the log viewer.
        action_open_log = Qg.QAction(
            Qg.QIcon.fromTheme("tools-report-bug"), "Report an issue...", self
        )
        action_open_log.triggered.connect(self.open_log_viewer)
        self.hamburger_menu.addAction(action_open_log)

        self.reload_stroke_width_icons()
        self.theme_is_dark_changed.connect(self.reload_stroke_width_icons)
        self.theme_is_dark_changed.connect(self.sketchpad.recolor_pen)
        self.theme_is_dark_changed.connect(self.load_training_symbol_data)

        if self.debug:
            # Add an intentional crash button.
            self.hamburger_menu.addSeparator()
            action = Qg.QAction("Simulate crash", self)
            action.triggered.connect(self.simulate_crash)
            self.hamburger_menu.addAction(action)

    def change_pen_width(self, width: int) -> None:
        """
        Change the pen width.
        """
        self.sketchpad.set_pen_width(width)
        self.config.stroke_width = width
        self.config.save()

    def reload_stroke_width_icons(self) -> None:
        """
        Reload the stroke width icons.
        """
        theme = "dark" if self.theme_is_dark.get() else "light"
        for action in self.stroke_width_menu.actions():
            action.setIcon(gu.load_custom_icon(f"stroke{action.width}", theme))  # noqa

    def open_log_viewer(self) -> None:
        logger.debug("Opening issue reporter.")
        issue_reporter = ird.IssueReporter(self)
        issue_reporter.exec()

    def open_symbol_list(self) -> None:
        """
        Open the symbol list.
        """
        if self.symbol_list is None:
            self.symbol_list = sl.SymbolList(self.symbols, self.similar_symbols)
            self.theme_is_dark_changed.connect(self.symbol_list.on_theme_change)
        self.symbol_list.show()

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

        # Delay it with a timer to wait for the palette to propagate.
        Qc.QTimer.singleShot(0, partial(self.theme_is_dark_changed.emit, self.theme_is_dark))

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
            Qc.QTimer.singleShot(0, self.show_predictions)

    def simulate_crash(self) -> None:
        """
        Simulate a crash by raising an exception.
        """
        raise Exception("This is a simulated crash.")

    # =========================================== Detection ==========================================

    def detect_symbol(self) -> None:
        # Get the sketch, tensorize it, and predict the symbol.
        start = time.time()
        strokes, _, _, _ = self.sketchpad.get_clean_strokes()
        logger.debug(f"Gathering strokes took {(time.time() - start) * 1000:.2f}ms")
        tensorized = ig.tensorize_strokes(strokes, trn.image_size)
        prediction = inf.predict(tensorized, self.model, self.label_decoder)
        self.current_predictions = prediction
        self.show_predictions()
        logger.debug(f"Prediction update took {(time.time() - start) * 1000:.2f}ms")

    def show_predictions(
        self,
    ) -> None:
        """
        Show the predictions in the result box.
        """
        print(83756329856429)

        # Empty out the widget containing the predictions.
        # self.widget_predictions.layout().deleteLater()
        # self.widget_predictions.setLayout(Qw.QVBoxLayout())
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                if widget := item.widget():
                    widget.deleteLater()  # If it's a widget, delete it
                elif sublayout := item.layout():
                    clear_layout(sublayout)  # If it's a layout, clear it recursively
                # Otherwise, it's a spacer item, no special handling needed
                del item  # Delete the layout item itself

        clear_layout(self.widget_predictions.layout())

        # print("=========================================")
        # for symbol, confidence in predictions:
        #     print(f"{symbol}: {confidence:.2f}")
        hex_color = self.palette().color(Qg.QPalette.Text).name()
        for symbol, confidence in self.current_predictions:
            # Craft a list item with the symbol and confidence.
            # This consists of a horizontal layout with an svg widget on the left,
            # and a vertical stack of labels on the right:
            # \usepackage{ thingy }    # if there is a package
            # \command
            # mode (confidence)

            # If this symbol has similar symbols, we want to display
            # them all together in a framed box.
            lookalikes = self.similar_symbols.get(symbol, None)
            stack = None
            if lookalikes:
                frame = Qw.QFrame()
                frame.setStyleSheet(
                    f"QFrame {{ background: {self.palette().color(Qg.QPalette.AlternateBase).name()}; }}"
                )
                stack = Qw.QVBoxLayout()
                frame.setLayout(stack)

            for s in [symbol] + list(lookalikes or []):
                symbol_data = self.symbols[s]
                outer_layout = Qw.QHBoxLayout()
                svg_widget = Qsw.QSvgWidget()
                svg_widget.load(ut.load_symbol_svg(symbol_data, hex_color))
                svg_widget.renderer().setAspectRatioMode(Qc.Qt.KeepAspectRatio)
                svg_widget.setFixedSize(64, 64)
                outer_layout.addWidget(svg_widget)

                inner_layout = Qw.QVBoxLayout()
                label_policy = Qw.QSizePolicy(Qw.QSizePolicy.Preferred, Qw.QSizePolicy.Minimum)
                if not symbol_data.package_is_default():
                    package_label = Qw.QLabel(f"\\usepackage{{ {symbol_data.package} }}")
                    package_label.setSizePolicy(label_policy)
                    package_label.setTextInteractionFlags(Qc.Qt.TextSelectableByMouse)
                    inner_layout.addWidget(package_label)
                command_label = Qw.QLabel(symbol_data.command)
                command_label.setTextInteractionFlags(Qc.Qt.TextSelectableByMouse)
                # Make this one 1.5 times bigger.
                font = command_label.font()
                font.setPointSize(int(font.pointSize() * 1.5))
                font.setBold(True)
                command_label.setSizePolicy(label_policy)
                command_label.setFont(font)
                inner_layout.addWidget(command_label)
                mode_label = Qw.QLabel(f"{symbol_data.mode_str()} (Match: {confidence:.1%})")
                mode_label.setTextInteractionFlags(Qc.Qt.TextSelectableByMouse)
                font = mode_label.font()
                font.setPointSize(int(font.pointSize() * 0.9))
                mode_label.setFont(font)
                mode_label.setSizePolicy(label_policy)
                inner_layout.setSpacing(0)
                inner_layout.setContentsMargins(0, 0, 0, 0)
                inner_layout.addWidget(mode_label)
                # Squish the inner layout together vertically.
                # Do this by making them not expand vertically, with a center alignment vertically.

                outer_layout.addLayout(inner_layout)

                if lookalikes:
                    stack.addLayout(outer_layout)
                else:
                    # Pad the outer layout with a left margin, so
                    # that it lines up with the framed lookalikes.
                    outer_layout.setContentsMargins(6, 0, 0, 0)
                    self.widget_predictions.layout().addLayout(outer_layout)
            if lookalikes:
                self.widget_predictions.layout().addWidget(frame)
        # Slap a spacer on the end to push the items to the top.
        self.widget_predictions.layout().addStretch()

    # =========================================== Training ===========================================

    def get_next_symbol(self) -> None:
        """
        Get the next symbol to draw.
        """
        # Check if the user specified a specific symbol.
        requested_symbol = self.lineEdit_train_symbol.text().strip()
        if requested_symbol:
            if requested_symbol in self.symbols:
                self.set_training_symbol(requested_symbol)
            else:
                gu.show_warning(
                    self,
                    "Invalid Symbol",
                    f"The symbol '{requested_symbol}' is not a valid symbol."
                    f"\n\nExample of a valid symbol key: {list(self.symbols.keys())[0]}",
                )

        else:
            bias = (
                self.horizontalSlider_selection_bias.value()
                / self.horizontalSlider_selection_bias.maximum()
            )
            new_symbol_key = self.data_recorder.select_symbol(bias)
            while self.current_symbol is not None and new_symbol_key == self.current_symbol.key:
                new_symbol_key = self.data_recorder.select_symbol(bias)

            self.set_training_symbol(new_symbol_key)

        self.submission_count = 1
        self.update_submission_count()

    def set_training_symbol(self, new_symbol_key: str) -> None:
        self.current_symbol = self.symbols[new_symbol_key]
        self.load_training_symbol_data()
        self.sketchpad.clear()

    def load_training_symbol_data(self) -> None:
        if self.current_symbol is None:
            return
        self.label_training_name.setText(self.current_symbol.command)
        self.label_symbol_rarity.setText(
            str(self.data_recorder.get_symbol_rarity(self.current_symbol.key))
        )
        self.label_symbol_samples.setText(
            str(self.data_recorder.get_symbol_sample_count(self.current_symbol.key))
        )

        hex_color = self.palette().color(Qg.QPalette.Text).name()
        self.widget_training_symbol.load(ut.load_symbol_svg(self.current_symbol, hex_color))
        self.widget_training_symbol.renderer().setAspectRatioMode(Qc.Qt.KeepAspectRatio)
        self.widget_training_symbol.setFixedSize(200, 200)

    def previous_symbol(self) -> None:
        """
        Go back to the previous symbol.
        """
        old_drawing = self.data_recorder.undo_submission()
        self.set_training_symbol(old_drawing.key)
        self.sketchpad.load_strokes(old_drawing)
        if self.submission_count > 1:
            self.submission_count -= 1
        self.update_submission_count()

    def submit_training_drawing(self) -> None:
        """
        Submit the drawing for training.
        """
        self.data_recorder.submit_drawing(
            st.SymbolDrawing(self.current_symbol.key, *self.sketchpad.get_clean_strokes())
        )
        self.sketchpad.clear()
        self.submission_count += 1

        max_submissions = self.spinBox_max_submissions.value()
        if self.submission_count > max_submissions:
            self.get_next_symbol()
        else:
            self.update_submission_count()
            self.load_training_symbol_data()

    def update_submission_count(self) -> None:
        """
        Update the submission count.
        """
        max_submissions = self.spinBox_max_submissions.value()
        self.label_submission_number.setText(f"{self.submission_count}/{max_submissions}")
