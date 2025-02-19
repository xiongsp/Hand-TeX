import argparse
import platform
import sys
from importlib import resources

import PySide6.QtGui as Qg
import PySide6.QtWidgets as Qw
from loguru import logger

import handtex.utils as ut
from handtex import __program__, __display_name__, __version__, __description__
import handtex.gui_utils as gu
import handtex.data.theme_icons as theme_icons_data
from handtex.driver_mainwindow import MainWindow
from handtex.utils import resource_path


def main() -> None:
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description=__description__, prog=__display_name__)

    # Add flags: debug and train and version
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--train", "-t", action="store_true", help="Create new training data")
    parser.add_argument("--version", "-v", action="version", version=f"{__program__} {__version__}")
    parser.add_argument(
        "--new-data-dir", "-n", help="Directory to store new training data", default=""
    )

    args = parser.parse_args()

    # Set up logging.
    logger.remove()
    ut.get_log_path().parent.mkdir(parents=True, exist_ok=True)

    # When bundling an executable, stdout can be None if no console is supplied.
    if sys.stdout is not None:
        if args.debug:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="WARNING")

    # Log up to 10MB to the log file.
    logger.add(str(ut.get_log_path()), rotation="10 MB", retention="1 week", level="DEBUG")

    # Set up a preliminary exception handler so that this still shows up in the log.
    # Once the gui is up and running it'll be replaced with a call to the gui's error dialog.
    def exception_handler(exctype, value, traceback) -> None:
        logger.opt(depth=1, exception=(exctype, value, traceback)).critical(
            "An uncaught exception was raised before the GUI was initialized."
        )

    sys.excepthook = exception_handler

    # Dump the system info.
    logger.info(ut.collect_system_info(__file__))

    # Dump the command line arguments if in debug mode.
    if args.debug:
        logger.debug(f"Launch arguments: {args}")

    # Start Qt runtime.
    app = Qw.QApplication(sys.argv)

    theme_icons = str(resource_path(theme_icons_data))

    Qg.QIcon.setFallbackSearchPaths([":/icons", theme_icons])
    # We need to set an initial theme on Windows, otherwise the icons will fail to load
    # later on, even when switching the theme again.
    if platform.system() != "Linux":
        Qg.QIcon.setThemeName("breeze")
        Qg.QIcon.setThemeSearchPaths([":/icons", theme_icons])

    try:
        window = MainWindow(args.debug, args.train, args.new_data_dir)
        window.show()
        sys.exit(app.exec())
    except Exception:
        gu.show_exception(None, "Failed to launch", "Failed to initialize the main window.")
    finally:
        logger.info(ut.SHUTDOWN_MESSAGE + "\n")


if __name__ == "__main__":
    main()
