import platform
import sys
import time
from functools import partial
from pathlib import Path

import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtSvgWidgets as Qsw
import PySide6.QtWidgets as Qw
from PySide6.QtCore import Signal
from loguru import logger
from PySide6.QtCore import Slot
import numpy as np

import handtex.config as cfg
import handtex.data_recorder as dr
import handtex.gui_utils as gu
import handtex.issue_reporter_driver as ird
import handtex.state_saver as ss
import handtex.structures as st
import handtex.symbol_list as sl
import handtex.symbol_relations as sr
import handtex.about_driver as ad
import handtex.utils as ut
import handtex.worker_thread as wt
from handtex.detector.image_gen import IMAGE_SIZE
import handtex.detector.inference as inf
import handtex.detector.model as mdl
from handtex import __display_name__, __version__
from handtex.ui_generated_files.ui_Mainwindow import Ui_MainWindow


class MainWindow(Qw.QMainWindow, Ui_MainWindow):
    config: cfg.Config = None
    debug: bool

    symbol_data: sr.SymbolData | None

    inference_queue: Qc.QThreadPool

    hamburger_menu: Qw.QMenu
    theming_menu: Qw.QMenu
    enabled_packages_menu: Qw.QMenu
    stroke_width_menu: Qw.QMenu

    about: ad.AboutWidget | None

    default_palette: Qg.QPalette
    default_style: str
    theme_is_dark: ut.Shared[bool]
    theme_is_dark_changed = Signal(bool)  # When true, the new theme is dark.

    state_saver: ss.StateSaver

    # Detection:
    model: mdl.CNN | None
    label_decoder: dict[int, str] | None
    current_predictions: list[tuple[str, float]]
    detection_menu_action: Qg.QAction | None

    # Training:
    data_recorder: dr.DataRecorder | None
    current_symbol: st.Symbol | None
    submission_count: int
    has_submission: Signal = Signal(bool)
    training_menu_action: Qg.QAction | None

    # Symbol list:
    symbol_list: sl.SymbolList | None

    def __init__(
        self,
        debug: bool,
        train: bool,
        new_data_dir: str,
    ) -> None:
        start = time.time()
        Qw.QMainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle(f"{__display_name__} {__version__}")
        self.setWindowIcon(gu.load_custom_icon("logo"))
        self.debug = debug

        self.current_predictions = []

        self.submission_count = 1
        self.current_symbol = None

        self.training_menu_action = None
        self.detection_menu_action = None

        self.data_recorder = None

        symbol_data_start = time.time()
        self.symbol_data = sr.SymbolData()
        logger.debug(f"Symbol data loaded in {(time.time() - symbol_data_start) * 1000:.2f}ms")

        self.config = self.load_config()
        # Set the new data directory if one is supplied.
        if new_data_dir:
            self.config.new_data_dir = new_data_dir
            self.config.save()
        self.config.pretty_log()

        self.theme_is_dark = ut.Shared[bool](True)

        self.initialize_ui()

        # Set up the thread pool for inference.
        # We want to limit this to one thread, so it acts as an
        # asynchronous dispatcher.
        self.inference_queue = Qc.QThreadPool()
        self.inference_queue.setMaxThreadCount(1)

        self.save_default_palette()
        self.load_config_theme()

        self.state_saver = ss.StateSaver("mainwindow")
        self.init_state_saver()

        self.sketchpad.new_drawing.connect(self.detect_symbol)
        self.theme_is_dark_changed.connect(self.show_predictions)

        self.model = None
        self.label_decoder = None
        self.start_model_loader()

        if train:
            self.switch_to_training()
        else:
            self.switch_to_classification()

        # Asynchronously load the symbol list.
        self.symbol_list = sl.SymbolList(self.symbol_data)
        self.theme_is_dark_changed.connect(self.symbol_list.on_theme_change)

        logger.debug(f"Initialization took {(time.time() - start) * 1000:.2f}ms")

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
        self.pushButton_browse_new_data_dir.clicked.connect(self.browse_new_data_dir2)
        self.lineEdit_new_data_dir.editingFinished.connect(self.update_new_data_dir)

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

        # Initialize the default split to 80:20.
        width1, width2 = self.splitter.sizes()
        logger.debug(f"Splitter sizes: {self.splitter.sizes()}")
        self.splitter.setSizes([(width1 + width2) * 0.8, (width1 + width2) * 0.2])
        logger.debug(f"Splitter sizes: {self.splitter.sizes()}")

        self.state_saver.restore()

    def closeEvent(self, event: Qg.QCloseEvent) -> None:
        """
        Notify config on close.
        """
        logger.info("Closing window.")
        self.state_saver.save()
        if self.inference_queue.activeThreadCount():
            # Process Qt events so that the message shows up.
            Qc.QCoreApplication.processEvents()
            self.inference_queue.waitForDone()

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
        self.hamburger_menu = Qw.QMenu()
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
            Qg.QIcon.fromTheme("edit-line-width"), "Line thickness"
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

        # Setting to scroll up when drawing.
        action_scroll_on_draw = Qg.QAction("Scroll back up when drawing", self)
        action_scroll_on_draw.setCheckable(True)
        action_scroll_on_draw.setChecked(self.config.scroll_on_draw)
        action_scroll_on_draw.triggered.connect(self.toggle_scroll_on_draw)
        self.hamburger_menu.addAction(action_scroll_on_draw)

        action_single_click_to_copy = Qg.QAction("Single click to copy", self)
        action_single_click_to_copy.setCheckable(True)
        action_single_click_to_copy.setChecked(self.config.single_click_to_copy)
        action_single_click_to_copy.triggered.connect(self.toggle_single_click_to_copy)
        self.hamburger_menu.addAction(action_single_click_to_copy)

        # Menu to select what packages to (not) exclude from predictions.
        self.enabled_packages_menu = self.hamburger_menu.addMenu(
            Qg.QIcon.fromTheme("package"), "Show additional matches"
        )
        package_list = self.symbol_data.packages.copy()
        package_list.remove("latex2e")  # We don't want to allow disabling the built-in stuff.
        package_action_group = Qg.QActionGroup(self)
        package_action_group.setExclusive(False)
        for package in package_list:
            action = Qg.QAction(package, self)
            action.setCheckable(True)
            action.setChecked(package not in self.config.disabled_packages)
            package_action_group.addAction(action)
            action.triggered.connect(partial(self.toggle_package, package))
            self.enabled_packages_menu.addAction(action)

        self.hamburger_menu.addSeparator()

        # About stuff.
        action_about = Qg.QAction(Qg.QIcon.fromTheme("help-about"), "About Hand TeX", self)
        action_about.triggered.connect(self.open_about)
        self.hamburger_menu.addAction(action_about)

        action_online_help = Qg.QAction(
            Qg.QIcon.fromTheme("internet-services"), "Online Documentation", self
        )
        action_online_help.triggered.connect(self.open_online_documentation)
        self.hamburger_menu.addAction(action_online_help)

        # Scan image.
        # This didn't work out, so it's disabled for now. The model just can't
        # handle the complexity of real-world images.
        # action_scan = Qg.QAction(Qg.QIcon.fromTheme("viewimage"), "Open Image", self)
        # action_scan.triggered.connect(self.browse_image)
        # self.hamburger_menu.addAction(action_scan)

        # Offer opening the log viewer.
        action_open_log = Qg.QAction(
            Qg.QIcon.fromTheme("tools-report-bug"), "Report an issue...", self
        )
        action_open_log.triggered.connect(self.open_log_viewer)
        self.hamburger_menu.addAction(action_open_log)

        # Offer to enter training mode.
        action_training = Qg.QAction(
            Qg.QIcon.fromTheme("draw-freehand"), "Help symbol training", self
        )
        action_training.triggered.connect(self.switch_to_training)
        self.hamburger_menu.addAction(action_training)
        self.training_menu_action = action_training
        # Offer way back to classification mode.
        action_classification = Qg.QAction(
            Qg.QIcon.fromTheme("search"), "Back to detection mode", self
        )
        action_classification.triggered.connect(self.switch_to_classification)
        self.hamburger_menu.addAction(action_classification)
        self.detection_menu_action = action_classification

        self.reload_stroke_width_icons()
        self.theme_is_dark_changed.connect(self.reload_stroke_width_icons)
        self.theme_is_dark_changed.connect(self.sketchpad.recolor)
        self.theme_is_dark_changed.connect(self.load_training_symbol_data)

        if self.debug:
            # Add an intentional crash button.
            self.hamburger_menu.addSeparator()
            action = Qg.QAction("Simulate crash", self)
            action.triggered.connect(self.simulate_crash)
            self.hamburger_menu.addAction(action)

    @staticmethod
    def open_online_documentation() -> None:
        """
        Open the online documentation in the default browser.
        """
        logger.debug("Opening online documentation.")
        Qg.QDesktopServices.openUrl(Qc.QUrl("https://github.com/VoxelCubes/Hand-TeX"))

    def open_about(self) -> None:
        """
        Open the about dialog.
        """
        logger.debug("Opening about dialog.")
        # Bodge in an instance variable to prevent garbage collection from immediately closing the window
        # due to not opening it modally.
        self.about = ad.AboutWidget(self)
        self.about.show()

    def switch_to_training(self) -> None:
        """
        Switch to training mode.
        """
        logger.info("Training mode active.")
        self.stackedWidget_right.setCurrentIndex(1)
        if self.data_recorder is None:
            self.data_recorder = dr.DataRecorder(
                self.symbol_data,
                self.has_submission,
                self.config.new_data_dir,
            )
            self.data_recorder.has_submissions.connect(self.pushButton_undo_submit.setEnabled)
        self.lineEdit_new_data_dir.setText(str(self.config.new_data_dir))
        self.training_menu_action.setVisible(False)
        self.detection_menu_action.setVisible(True)
        self.pushButton_submit.setShortcutEnabled(True)
        self.get_next_symbol()

    def switch_to_classification(self) -> None:
        """
        Switch to classification mode.
        """
        logger.info("Classification mode active.")
        self.stackedWidget_right.setCurrentIndex(0)
        self.training_menu_action.setVisible(True)
        self.detection_menu_action.setVisible(False)
        self.pushButton_submit.setShortcutEnabled(False)
        self.sketchpad.clear()

    def in_detection_mode(self) -> bool:
        return self.stackedWidget_right.currentIndex() == 0

    def change_pen_width(self, width: int) -> None:
        """
        Change the pen width.
        """
        self.sketchpad.set_pen_width(width)
        self.config.stroke_width = width
        self.config.save()

    def toggle_scroll_on_draw(self) -> None:
        """
        Toggle the scroll on draw setting.
        """
        self.config.scroll_on_draw = not self.config.scroll_on_draw
        self.config.save()

    def toggle_single_click_to_copy(self) -> None:
        """
        Toggle the single click to copy setting.
        """
        self.config.single_click_to_copy = not self.config.single_click_to_copy
        self.config.save()

    def toggle_package(self, package: str) -> None:
        """
        Toggle the enabled state of a package.
        """
        if package in self.config.disabled_packages:
            self.config.disabled_packages.remove(package)
        else:
            self.config.disabled_packages.append(package)
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
        self.symbol_list.show()

    def browse_new_data_dir2(self) -> None:
        """
        Browse for a new data directory.
        """
        logger.debug("Browsing for new data directory.")
        new_data_dir = Qw.QFileDialog.getExistingDirectory(
            self, "Select a new data directory", str(self.config.new_data_dir)
        )
        if new_data_dir:
            self.lineEdit_new_data_dir.setText(new_data_dir)
            self.update_new_data_dir()

    def update_new_data_dir(self) -> None:
        """
        Read the current new data directory from the line edit.
        """
        new_data_dir = self.lineEdit_new_data_dir.text()
        self.config.new_data_dir = new_data_dir
        self.config.save()
        # Because we can only trigger this event through interaction with
        # the training mode UI, we can safely assume that the data recorder
        # was initialized.
        self.data_recorder.set_new_data_dir(Path(new_data_dir))

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

    def start_model_loader(self) -> None:
        """
        Start a worker thread to load the model.
        """
        worker = wt.Worker(
            inf.load_model_and_decoder,
            ut.get_model_path(),
            ut.get_encodings_path(),
            no_progress_callback=True,
        )
        worker.signals.result.connect(self.model_loader_result)
        worker.signals.error.connect(self.model_loader_error)
        self.inference_queue.start(worker)

    def model_loader_result(self, result) -> None:
        """
        The model was loaded without errors.
        We just need to make sure that the model is properly loaded into vram,
        so we do a dry-run prediction to touch the gpu.
        """
        self.model, self.label_decoder = result
        logger.info("Model loaded. Proceeding to cold-start the model.")
        self.start_detection([[(0, 0)]], dry_run=True)

    @Slot(wt.WorkerError)
    def model_loader_error(self, error: wt.WorkerError) -> None:
        gu.show_exception(
            self,
            self.tr("Model Loading Failed"),
            self.tr("Failed to load the model from {mpath} or the encodings from {epath}").format(
                mpath=ut.get_model_path(), epath=ut.get_encodings_path()
            ),
            error,
        )

    def detect_symbol(self) -> None:
        """
        Gather the strokes from the sketchpad and start the detection process.
        """
        if not self.in_detection_mode():
            return
        if self.model is None:
            logger.error("Model is not loaded yet, skipping prediction.")
            return
        # Get the sketch and predict the symbol.
        # start = time.time()
        strokes, _, _, _ = self.sketchpad.get_clean_strokes()
        # logger.debug(f"Got {len(strokes)} strokes in {(time.time() - start) * 1000:.2f}ms.")
        self.start_detection(strokes)

    def start_detection(self, strokes: list[list[tuple[int, int]]], dry_run: bool = False) -> None:
        """
        Start the detection process.

        :param strokes: The strokes to detect.
        :param dry_run: If true, don't show the predictions. This is to prevent a cold-start on GPU.
        """
        worker = wt.Worker(
            inf.predict_strokes, strokes, self.model, self.label_decoder, no_progress_callback=True
        )
        if not dry_run:
            worker.signals.result.connect(self.detection_result)
        worker.signals.error.connect(self.detection_error)
        self.inference_queue.start(worker)

    def start_detection_image(self, image_data: np.ndarray, dry_run: bool = False) -> None:
        """
        Start the detection process.
        This one works for numpy image data.

        :param image_data: The image to detect.
        :param dry_run: If true, don't show the predictions. This is to prevent a cold-start on GPU.
        """
        worker = wt.Worker(
            inf.predict_image, image_data, self.model, self.label_decoder, no_progress_callback=True
        )
        if not dry_run:
            worker.signals.result.connect(self.detection_result)
        worker.signals.error.connect(self.detection_error)
        self.inference_queue.start(worker)

    def detection_result(self, result: list[tuple[str, float]]) -> None:
        self.current_predictions = result
        self.show_predictions()
        if self.config.scroll_on_draw:
            self.scrollArea_predictions.scroll_to_top()

    def detection_error(self, error: wt.WorkerError) -> None:
        gu.show_exception(
            self,
            self.tr("Detection Failed"),
            self.tr("Failed to detect the symbol."),
            error,
        )

    def show_predictions(
        self,
    ) -> None:
        """
        Show the predictions in the result box.
        """

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
            similarity_group = list(self.symbol_data.get_similarity_group(symbol))
            # Extract the set of packages represented by the similarity group.
            packages_represented = {self.symbol_data[s].package for s in similarity_group}
            # Remove disabled packages as long as doing so leaves at least one package.
            while len(packages_represented) > 1:
                for group in list(
                    packages_represented
                ):  # iterate over a copy to avoid modifying the set during iteration
                    if group in self.config.disabled_packages:
                        packages_represented.remove(group)
                        break
                else:
                    break
            # Apply filtering to the similarity group.
            similarity_group = [
                s for s in similarity_group if self.symbol_data[s].package in packages_represented
            ]

            similarity_stack = None
            frame = None
            if len(similarity_group) > 1:
                frame = Qw.QFrame()
                frame.setStyleSheet(
                    f"QFrame {{ background: {self.palette().color(Qg.QPalette.AlternateBase).name()}; }}"
                )
                similarity_stack = Qw.QVBoxLayout()
                # Set spacing between items in the layout.
                similarity_stack.setSpacing(12)
                frame.setLayout(similarity_stack)

            for s in similarity_group:
                prediction_widget = PredictionWidget(
                    self, self.symbol_data[s], confidence, hex_color, self.debug
                )
                prediction_widget.symbol_for_clipboard.connect(self.copy_symbol_to_clipboard)

                if similarity_stack:
                    similarity_stack.addWidget(prediction_widget)
                else:
                    # Pad the outer layout with a left margin, so
                    # that it lines up with the framed lookalikes.
                    prediction_widget.setContentsMargins(6, 0, 0, 0)
                    self.widget_predictions.layout().addWidget(prediction_widget)
            if similarity_stack:
                self.widget_predictions.layout().addWidget(frame)
        # Slap a spacer on the end to push the items to the top.
        self.widget_predictions.layout().addStretch()

    @Slot(str, bool)
    def copy_symbol_to_clipboard(self, clipboard_text: str, is_double_click: bool) -> None:
        """
        Copy the symbol to the clipboard and open a toast at the bottom
        of the scroll area.
        """
        if is_double_click == self.config.single_click_to_copy:
            return
        clipboard = Qw.QApplication.clipboard()
        clipboard.setText(clipboard_text)
        toast = Toast(self.scrollArea_predictions, f"Copied {clipboard_text} to clipboard.")
        toast.show()

    # ======================================= Loading Images =======================================

    def browse_image(self) -> None:
        """
        Load an image from the filesystem.
        """
        image_path, _ = Qw.QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.home()), "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if not image_path:
            return
        self.load_image_from_path(image_path)

    def load_image_from_path(self, image_path: str) -> None:
        """
        Load an image from the filesystem.
        """
        logger.info(f"Loading image from {image_path}.")
        image = Qg.QImage(image_path)
        if image.isNull():
            gu.show_warning(self, "Invalid Image", "The selected image is not valid.")
            return
        # Scale the image down to fit the canvas size and then run inference.
        scaled_image = image.scaled(
            Qc.QSize(IMAGE_SIZE, IMAGE_SIZE), Qc.Qt.KeepAspectRatio, Qc.Qt.SmoothTransformation
        )
        # Assuming 'scaled_image' is a QImage object
        if scaled_image.format() != Qg.QImage.Format_Grayscale8:
            grayscale_image = scaled_image.convertToFormat(Qg.QImage.Format_Grayscale8)
        else:
            grayscale_image = scaled_image

        # Convert QImage to a NumPy array directly
        width = grayscale_image.width()
        height = grayscale_image.height()
        image_data = np.array(grayscale_image.constBits()).reshape((height, width))
        # Snap the image values to 0 and 255, threshold 127.
        image_data = np.where(image_data > 127, 255, 0).astype(np.uint8)
        # # debug-render the image
        # import matplotlib.pyplot as plt
        #
        # plt.imshow(image_data, cmap="gray")
        # plt.show()

        # Convert the image to 1-channel 8-bit grayscale.
        self.start_detection_image(image_data)

    # =========================================== Training ===========================================

    def get_next_symbol(self) -> None:
        """
        Get the next symbol to draw.
        """
        # Check if the user specified a specific symbol.
        requested_symbol = self.lineEdit_train_symbol.text().strip()
        if requested_symbol:
            if requested_symbol in self.symbol_data:
                self.set_training_symbol(requested_symbol)
            else:
                response = gu.show_question(
                    self,
                    "Unknown Symbol",
                    f"The symbol '{requested_symbol}' is not a recognized symbol."
                    f"\n\nExample of a valid symbol key: {self.symbol_data.leaders[0]}"
                    "\n\nDo you want to train this symbol anyway?",
                )
                if response == Qw.QMessageBox.Yes:
                    self.set_training_symbol(requested_symbol)

        else:
            bias = (
                self.horizontalSlider_selection_bias.value()
                / self.horizontalSlider_selection_bias.maximum()
            )
            new_symbol_key = self.data_recorder.select_symbol(bias)
            while (
                (self.current_symbol is not None and new_symbol_key == self.current_symbol.key)
                or new_symbol_key in self.data_recorder.last_100_symbols
                or self.data_recorder.get_symbol_rarity(new_symbol_key) < (100 * bias - 50)
            ):
                new_symbol_key = self.data_recorder.select_symbol(bias)

            self.data_recorder.last_100_symbols.append(new_symbol_key)
            if len(self.data_recorder.last_100_symbols) > 100:
                self.data_recorder.last_100_symbols.pop(0)
            self.set_training_symbol(new_symbol_key)

        self.submission_count = 1
        self.update_submission_count()

    def set_training_symbol(self, new_symbol_key: str) -> None:
        if new_symbol_key in self.symbol_data:
            self.current_symbol = self.symbol_data[new_symbol_key]
        else:
            self.current_symbol = st.Symbol.dummy(new_symbol_key)
        self.load_training_symbol_data()
        self.sketchpad.clear()

    def load_training_symbol_data(self) -> None:
        if self.current_symbol is None:
            return
        self.label_training_name.setText(self.current_symbol.command)
        self.label_symbol_rarity.setText(
            f"{self.data_recorder.get_symbol_rarity(self.current_symbol.key)}%"
        )
        self.label_symbol_samples.setText(
            str(self.data_recorder.get_symbol_sample_count(self.current_symbol.key))
        )

        # Check if the symbol filename is missing, and if so, insert the varnothing symbol.
        # This can happen particularly when a new symbol is being trained that doesn't yet exist.
        if not self.current_symbol.filename:
            symbol_nothing = self.symbol_data["amssymb-_varnothing"]
            symbol_obj = symbol_nothing
        else:
            symbol_obj = self.current_symbol

        hex_color = self.palette().color(Qg.QPalette.Text).name()
        self.widget_training_symbol.load(ut.load_symbol_svg(symbol_obj, hex_color))
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
        if self.sketchpad.is_empty():
            gu.show_warning(self, "Empty Drawing", "You cannot submit an empty drawing.")
            return
        self.data_recorder.submit_drawing(
            st.SymbolDrawing(
                self.current_symbol.key, *self.sketchpad.get_clean_strokes(simplify=True)
            )
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


class PredictionWidget(Qw.QWidget):
    # Signal to emit the symbol command when double clicked
    symbol_for_clipboard = Signal(str, bool)

    def __init__(
        self,
        parent: Qw.QWidget,
        symbol_data: st.Symbol,
        confidence: float,
        background_color: str,
        debug: bool = False,
    ):
        super().__init__(parent)
        self.symbol_data = symbol_data
        self.debug = debug

        outer_layout = Qw.QHBoxLayout()
        outer_layout.setSpacing(6)

        # SVG widget
        svg_widget = Qsw.QSvgWidget()
        svg_widget.load(ut.load_symbol_svg(symbol_data, background_color))
        svg_widget.renderer().setAspectRatioMode(Qc.Qt.KeepAspectRatio)
        svg_widget.setFixedSize(64, 64)
        outer_layout.addWidget(svg_widget)

        # Right side (text) layout
        inner_layout = Qw.QVBoxLayout()
        label_policy = Qw.QSizePolicy(Qw.QSizePolicy.Preferred, Qw.QSizePolicy.Minimum)

        # Optional \usepackage label
        if not symbol_data.package_is_default():
            package_label = Qw.QLabel(f"\\usepackage{{{symbol_data.package}}}")
            package_label.setSizePolicy(label_policy)
            # package_label.setTextInteractionFlags(Qc.Qt.TextSelectableByMouse)
            inner_layout.addWidget(package_label)

        # Command label
        command_label = Qw.QLabel(symbol_data.command)
        # command_label.setTextInteractionFlags(Qc.Qt.TextSelectableByMouse)
        font = command_label.font()
        font.setPointSize(int(font.pointSize() * 1.5))
        font.setBold(True)
        command_label.setFont(font)
        command_label.setSizePolicy(label_policy)
        inner_layout.addWidget(command_label)

        # Mode + confidence label
        mode_label = Qw.QLabel(f"{symbol_data.mode_str()} (Match: {confidence:.1%})")
        # mode_label.setTextInteractionFlags(Qc.Qt.TextSelectableByMouse)
        font = mode_label.font()
        font.setPointSize(int(font.pointSize() * 0.9))
        mode_label.setFont(font)
        mode_label.setSizePolicy(label_policy)
        inner_layout.setSpacing(0)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(mode_label)

        # Combine
        outer_layout.addLayout(inner_layout)
        self.setLayout(outer_layout)

    def mouseDoubleClickEvent(self, event: Qg.QMouseEvent) -> None:
        """
        Emit the symbol command string when the widget is double clicked
        (including if the double click happens on a child widget).
        """
        if event.button() == Qc.Qt.MouseButton.LeftButton:
            self.symbol_for_clipboard.emit(self.symbol_data.command, True)
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: Qg.QMouseEvent) -> None:
        """
        Emit the symbol command string when the widget is clicked
        (including if the click happens on a child widget).
        """
        if event.button() == Qc.Qt.MouseButton.LeftButton:
            self.symbol_for_clipboard.emit(self.symbol_data.command, False)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: Qg.QContextMenuEvent) -> None:
        """
        Show a context menu on right click offering:
        - Copy Command
        - Copy Usepackage Snippet
        - Copy Key (only if debug=True)
        """
        menu = Qw.QMenu(self)

        # Always add "Copy Command"
        copy_command_action = menu.addAction("Copy Command")

        # Add "Copy UsePackage Snippet" only if a non-default package is used
        copy_usepackage_action = None
        if not self.symbol_data.package_is_default():
            copy_usepackage_action = menu.addAction("Copy UsePackage Snippet")

        # Add "Copy Key" only if debug is True
        copy_key_action = None
        if self.debug:
            copy_key_action = menu.addAction("Copy Key")

        chosen_action = menu.exec_(self.mapToGlobal(event.pos()))
        text_to_copy = ""
        if chosen_action == copy_command_action:
            text_to_copy = self.symbol_data.command

        elif chosen_action == copy_usepackage_action:
            text_to_copy = f"\\usepackage{{{self.symbol_data.package}}}"

        elif chosen_action == copy_key_action:
            text_to_copy = self.symbol_data.key

        if text_to_copy:
            # Send it with the double click flag set to true and to false, one of them
            # will work.
            self.symbol_for_clipboard.emit(text_to_copy, False)
            self.symbol_for_clipboard.emit(text_to_copy, True)


class Toast(Qw.QLabel):
    """A floating toast message for QScrollArea that auto-closes after 5 seconds."""

    _active_toasts = []

    def __init__(self, parent: Qw.QScrollArea, message: str):
        super().__init__(parent)

        # Close any other active toasts
        for toast in list(self._active_toasts):
            toast.close()
        self._active_toasts.clear()
        self._active_toasts.append(self)

        # Increase font size a bit to make the brief message more readable.
        font = self.font()
        font.setPointSize(round(font.pointSize() * 1.2))
        self.setFont(font)

        self.setText(message)
        self.setWordWrap(True)
        self.setAlignment(Qc.Qt.AlignCenter)
        self.setWindowFlag(Qc.Qt.FramelessWindowHint)
        self.setAttribute(Qc.Qt.WA_TransparentForMouseEvents, True)

        # Derive background color from parent's palette, invert its luminosity.
        self._apply_inverted_bg_and_text()

        # Close automatically after 5 seconds
        Qc.QTimer.singleShot(5000, self.close)

    def _apply_inverted_bg_and_text(self) -> None:
        """Derive the parent's background color, invert it, and pick a good text color."""
        # Get the scroll area viewport’s palette/color
        palette = self.parentWidget().viewport().palette()
        original_color = palette.color(Qg.QPalette.Window)

        # Convert color to HSV and invert V (luminosity)
        h, s, v, a = original_color.getHsv()
        # Cut the saturation in half to make the text color more readable.
        s = s // 2

        # invert the luminosity:
        v = 255 - v  # if v is in [0..255]

        # Pick text color: if background is dark, text is white, else black
        if v < 128:
            text_color = "white"
        else:
            text_color = "black"

        # Construct new color with same hue and saturation, inverted luminosity
        inverted_bg = Qg.QColor()
        inverted_bg.setHsv(h, s, v, a)

        # Use partial transparency so we see "through" the toast
        inverted_bg.setAlpha(240)

        # Apply styling
        self.setStyleSheet(
            f"""
            background-color: {inverted_bg.name(Qg.QColor.HexArgb)};
            color: {text_color};
            padding: 8px;
            border-radius: 5px;
        """
        )

    def showEvent(self, event):
        super().showEvent(event)

        # Use ~80% of the scroll area’s width so we don’t end up with a tall/narrow label
        viewport_rect = self.parentWidget().viewport().rect()
        max_width = int(viewport_rect.width() * 0.8)
        self.setFixedWidth(max_width)

        # Now that we have a set width, let the label expand as tall as needed
        self.adjustSize()

        # Position near bottom center of the scroll area's viewport
        x = (viewport_rect.width() - self.width()) // 2
        y = viewport_rect.height() - self.height() - 20
        self.move(x, y)
        self.raise_()

    def closeEvent(self, event):
        # Remove from active toasts if it’s still in the list
        try:
            self._active_toasts.remove(self)
        except ValueError:
            pass
        super().closeEvent(event)
