import difflib
import re
import os
import platform
import shutil
import sys
from collections import defaultdict
from importlib import resources
from io import StringIO
from pathlib import Path
from typing import get_type_hints, Generic, TypeVar, Optional

import PySide6
import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtWidgets as Qw
import psutil
from loguru import logger
from xdg import XDG_CONFIG_HOME, XDG_CACHE_HOME

import handtex.data
import handtex.structures as st
from handtex import __program__, __version__
from handtex.data import color_themes
from handtex.data import symbols
from handtex.data import symbol_metadata
import handtex.data.model


T = TypeVar("T")


class Shared(Generic[T]):
    def __init__(self, initial_value: Optional[T] = None) -> None:
        self._container = {"data": initial_value}

    def get(self) -> Optional[T]:
        return self._container["data"]

    def set(self, value: T) -> None:
        self._container["data"] = value

    def is_none(self) -> bool:
        return self._container["data"] is None


# Logging session markers.
STARTUP_MESSAGE = "---- Starting up ----"
SHUTDOWN_MESSAGE = "---- Shutting down ----"


def running_in_flatpak() -> bool:
    return Path("/.flatpak-info").exists()


def collect_system_info(callers_file: str) -> str:
    buffer = StringIO()
    buffer.write("\n" + STARTUP_MESSAGE)
    buffer.write("\n- Program Information -\n")
    buffer.write(f"Program: {__program__} {__version__}\n")
    buffer.write(f"Executing from: {callers_file}\n")
    buffer.write(f"Log file: {get_log_path()}\n")
    buffer.write(f"Config file: {get_config_path()}\n")
    buffer.write(f"Cache directory: {get_cache_path()}\n")
    buffer.write("- System Information -\n")
    buffer.write(f"Operating System: {platform.system()} {platform.release()}\n")
    if platform.system() == "Linux":
        buffer.write(f"Desktop Environment: {os.getenv('XDG_CURRENT_DESKTOP', 'unknown')}\n")
    if running_in_flatpak():
        buffer.write("Sandbox: Running in Flatpak\n")
    buffer.write(f"Machine: {platform.machine()}\n")
    buffer.write(f"Python Version: {sys.version}\n")
    buffer.write(f"PySide (Qt) Version: {PySide6.__version__}\n")
    buffer.write(f"Available Qt Themes: {', '.join(Qw.QStyleFactory.keys())}\n")
    current_app_theme = Qw.QApplication.style()
    current_app_theme_name = (
        current_app_theme.objectName() if current_app_theme else "System Default"
    )
    buffer.write(f"Current Qt Theme: {current_app_theme_name}\n")
    icon_theme_name = Qg.QIcon.themeName()
    icon_theme_name = icon_theme_name if icon_theme_name else "System Default"
    buffer.write(f"Current Icon Theme: {icon_theme_name}\n")
    buffer.write(
        f"Available Color Themes: {', '.join(map(lambda a: a[1], get_available_themes()))}\n"
    )
    buffer.write(f"System locale: {Qc.QLocale.system().name()}\n")
    buffer.write(f"CPU Cores: {os.cpu_count()}\n")
    buffer.write(f"Memory: {sys_virtual_memory_total() / 1024 ** 3:.2f} GiB\n")
    buffer.write(f"Swap: {sys_swap_memory_total() / 1024 ** 3:.2f} GiB\n")

    return buffer.getvalue()


def get_config_path() -> Path:
    """
    Get the path to the configuration file for both Linux and Windows.
    """
    xdg_path = os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config"

    if platform.system() == "Linux":
        path = Path(XDG_CONFIG_HOME, __program__, __program__ + "rc")
    elif platform.system() == "Windows":
        path = Path(
            xdg_path if "XDG_CONFIG_HOME" in os.environ else os.getenv("APPDATA"),
            __program__,
            __program__ + "config.ini",
        )
    elif platform.system() == "Darwin":
        path = Path(
            (
                xdg_path
                if "XDG_CONFIG_HOME" in os.environ
                else (Path.home() / "Library" / "Application Support")
            ),
            __program__,
            __program__ + "config.ini",
        )
    else:  # ???
        raise NotImplementedError("Your OS is currently not supported.")

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_path() -> Path:
    """
    Get the default suggested path to the cache directory for both Linux and Windows.
    """
    xdg_path = os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache"

    if platform.system() == "Linux":
        path = Path(XDG_CACHE_HOME, __program__)
    elif platform.system() == "Windows":
        path = Path(
            xdg_path if "XDG_CACHE_HOME" in os.environ else os.getenv("APPDATA"),
            __program__,
            "cache",
        )
    elif platform.system() == "Darwin":
        path = Path(
            xdg_path if "XDG_CACHE_HOME" in os.environ else (Path.home() / "Library" / "Caches"),
            __program__,
        )
    else:  # ???
        raise NotImplementedError("Your OS is currently not supported.")

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_log_path() -> Path:
    """
    Get the path to the log file.
    Use the cache directory for this.
    """
    return get_cache_path() / f"{__program__}.log"


def get_available_themes() -> list[tuple[str, str]]:
    """
    Check the data/color_themes directory for available themes.
    The theme name is the plain file name. The display name is either defined in the
    theme file under section General, key name.
    If not defined, the display name is the theme name but capitalized and
    with spaces instead of underscores.

    Note: The implicit system theme is not included in the list.

    :return: A list of available theme names with their display names.
    """
    # Simply discover all files in the themes folder.
    themes = []
    with resources.path(color_themes, "") as theme_dir:
        theme_dir = Path(theme_dir)
    for theme_file in theme_dir.iterdir():
        # Skip dirs and empty files.
        if theme_file.is_dir() or theme_file.stat().st_size == 0:
            continue
        theme_name = theme_file.stem
        content = theme_file.read_text()
        display_name = theme_name.replace("_", " ").capitalize()
        in_general_section = False
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("[General]"):
                in_general_section = True
            elif line.startswith("[") and line.endswith("]"):
                if in_general_section:
                    # We found general, but came across the next section now.
                    break
                in_general_section = False
            elif "=" in line:
                key, value = map(str.strip, line.split("=", 1))
                if key == "Name":
                    display_name = value
                    break
        themes.append((theme_name, display_name))

    return themes


def closest_match(word: str, choices: list[str]) -> str | None:
    """
    Return the closest match for the given word in the list of choices.
    If no good match is found, return None.
    """
    if word in choices:
        return word
    else:
        # Find the closest match using difflib:
        closest = difflib.get_close_matches(word, choices, 1, 0.5)  # 0.6 is the default threshold
        if closest:
            return str(closest[0])
        else:
            return None


def f_plural(value, singular: str, plural: str = "") -> str:
    """
    Selects which form to use based on the value.

    :param value: Value to check.
    :param singular: Singular form.
    :param plural: (Optional) Plural form. If not given, the singular form is used with an 's' appended.
    :return: The appropriate form.
    """
    if not plural:
        plural = singular + "s"
    return singular if value == 1 else plural


def f_time(seconds: int) -> str:
    """
    Format a time in seconds to a human readable string.
    Return a format like:
    1 second
    2 minutes 3 seconds
    4 hours 5 minutes
    """
    if seconds < 60:
        return f"{seconds} {f_plural(seconds, 'second')}"
    elif seconds < 60 * 60:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes} {f_plural(minutes, 'minute')} {seconds} {f_plural(seconds, 'second')}"
    else:
        hours = seconds // (60 * 60)
        minutes = (seconds % (60 * 60)) // 60
        return (
            f"{hours} {f_plural(hours, 'hour')} "
            f"{minutes} {f_plural(minutes, 'minute')}"
            f"   [You're batshit insane!]"
        )


def open_file(path: Path) -> None:
    """
    Open any given file with the default application.
    """
    logger.info(f"Opening file {path}")
    try:
        # Use Qt to open the file, so that it works on all platforms.
        Qg.QDesktopServices.openUrl(Qc.QUrl.fromLocalFile(str(path)))
    except Exception as e:
        logger.exception(e)


def ensure_unique_file_path(file_path: Path) -> Path:
    """
    Ensure that the file path is unique.
    If the file already exists, append a number to the file name,
    incrementing it until a unique file path is found.
    """
    counter = 1
    output_file_path = file_path
    while output_file_path.exists():
        output_file_path = file_path.parent / (
            file_path.stem + "_" + str(counter) + file_path.suffix
        )
        counter += 1
    return output_file_path


def backup_file(path: Path, extension: str = ".backup") -> Path:
    """
    Create a backup of the file by copying it to the same location with the given extension.
    """
    backup_path = path.with_suffix(path.suffix + extension)
    backup_path = ensure_unique_file_path(backup_path)
    logger.info(f"Backing up file {path} to {backup_path}")
    shutil.copy(path, backup_path)
    return backup_path


class RecoverableParseException(Exception):
    """
    This serves to wrap any exceptions that occur during parsing,
    so that additional info about the file can be included.
    These exceptions could be recovered from.
    """

    pass


class CriticalParseError(Exception):
    """
    This serves to wrap any critical errors that occur during parsing,
    so that additional info about the file can be included.
    A critical error implies a failure to parse the file at all.
    """

    pass


def load_dict_to_attrs_safely(
    dataclass: object,
    data: dict,
    *,
    skip_attrs: list[str] | None = None,
    include_until_base: type | list[type] | None = None,
) -> list[RecoverableParseException]:
    """
    Load a dictionary into an attrs class while ensuring types are correct.
    Any type issues are logged and returned as a list of exceptions.
    If no exceptions are returned, the loading was successful.
    In the worst case, the object is simply left unchanged.
    When you have a dataclass that inherits from another one, type annotations won't be inherited,
    so set include_until_base to the base class to include all attributes up to (and including)
    that class (multiple inheritance is supported).

    :param dataclass: The dataclass to load the dictionary into.
    :param data: The dictionary to load.
    :param skip_attrs: [Optional] A list of attributes to skip.
    :param include_until_base: [Optional] Include attributes until this base class.
    :return: A list of exceptions that occurred during loading.
    """
    recoverable_exceptions: list[RecoverableParseException] = []
    type_info = get_type_hints(dataclass)
    # Gather type hints from base classes if requested.
    if include_until_base:
        if not isinstance(include_until_base, list):
            include_until_base = [include_until_base]
        base_classes = list(type(dataclass).__bases__)
        while base_classes:
            base_class = base_classes.pop(0)
            type_info.update(get_type_hints(base_class))
            if base_class not in include_until_base:
                base_classes.extend(list(base_class.__bases__))

    for attribute in type_info:
        if skip_attrs and attribute in skip_attrs:
            continue

        if attribute in data:
            # Attempt to coerce the type to the correct one.
            value = data[attribute]
            expected_type = type_info[attribute]
            try:
                setattr(dataclass, attribute, expected_type(value))
            except Exception as e:
                logger.exception(f"Failed to cast attribute {attribute} to the correct type.")
                recoverable_exceptions.append(
                    RecoverableParseException(
                        f"{type(e).__name__}: Failed to cast attribute {attribute} to the correct type: {e}"
                    )
                )

    return recoverable_exceptions


def resource_path(module, resource=""):
    return resources.as_file(resources.files(module).joinpath(resource))


def load_symbols() -> dict[str, st.Symbol]:
    """
    Load the symbols from the symbols.json file.

    :return: A dictionary of key to symbols.
    """

    with resources.path(handtex.data, "symbols.json") as symbols_file:
        symbol_list = st.Symbol.from_json(symbols_file)
    return {symbol.key: symbol for symbol in symbol_list}


def load_symbol_svg(symbol: st.Symbol, fill_color: str = "#000000") -> Qc.QByteArray:
    """
    Load the SVG for the given symbol key, applying the new fill color.
    The raw svg data is returned as a QByteArray, ready for a QSvgRenderer.

    :param symbol: The symbol to load.
    :param fill_color: The new fill color.
    :return: The raw SVG data.
    """
    with resources.path(symbols, f"{symbol.filename}.svg") as svg_file:
        svg_data = svg_file.read_text()

    # Recolor the SVG data.
    if fill_color == "#000000":
        return Qc.QByteArray(svg_data.encode("utf-8"))

    svg_data = svg_data.replace('stroke="#000000"', f'stroke="{fill_color}"')
    svg_data = svg_data.replace('stroke="#000"', f'stroke="{fill_color}"')
    svg_data = svg_data.replace('fill="#000000"', f'fill="{fill_color}"')

    return Qc.QByteArray(svg_data.encode("utf-8"))


def select_leader_symbols(
    symbol_keys: list[str], lookalikes: dict[str, tuple[str, ...]]
) -> list[str]:
    """
    Select the leader symbols from the list of symbol keys.
    Leaders either have no lookalikes or are the first symbol among a set of lookalikes.

    :param symbol_keys: List of symbol keys.
    :param lookalikes: Dictionary mapping symbol keys to sets of lookalike symbol keys.
    :return: List of leader symbols.
    """
    # Iterate over the lookalike's keys first, to ensure that
    # the first symbol in the lookalike set is always the leader.
    # Otherwise, if we iterated over all symbols, which have random order,
    # a different symbol might appear first and be selected as the leader.
    leaders = []
    for key in lookalikes:
        if not any(k in leaders for k in lookalikes[key]):
            leaders.append(key)
    for key in symbol_keys:
        if key not in leaders and key not in lookalikes:
            leaders.append(key)

    return leaders


def load_symbol_metadata_similarity() -> dict[str, tuple[str, ...]]:
    """
    Load the metadata for symbols.
    File format: each line contains a space separated list of symbol keys.

    :return: A dictionary mapping symbol keys to sets of symbol keys.
    """
    with resources.path(symbol_metadata, "") as metadata_dir:
        metadata_dir = Path(metadata_dir)
    files = list(metadata_dir.glob("similar*"))

    symbol_map = {}
    for file in files:
        with file.open("r") as f:
            for line in f:
                similar_keys = line.strip().split()
                for key in similar_keys:
                    if key not in symbol_map:
                        symbol_map[key] = tuple(k for k in similar_keys if k != key)
                    else:
                        logger.error(f"Duplicate symbol key found: {key}")

    return symbol_map


def load_symbol_metadata_self_symmetry() -> dict[str, list[st.Symmetry]]:
    """
    Load the metadata for symbol self-symmetric transformations.

    File format:
    symbol_key: symmetry1 symmetry2 symmetry3 ...

    :return: A dictionary mapping symbol keys to lists of symmetries.
    """
    with resource_path(symbol_metadata, "symmetry_self.txt") as file_path:
        with open(file_path, "r") as file:
            lines = file.readlines()

    symmetries = {}
    for line in lines:
        parts = line.strip().split()
        key = parts[0][:-1]  # Remove the colon at the end.
        symmetries[key] = [st.Symmetry(sym) for sym in parts[1:]]
    return symmetries


def load_symbol_metadata_other_symmetry() -> dict[str, list[tuple[str, list[st.Symmetry]]]]:
    """
    Load the metadata for symbol other-symmetric transformations.
    The file must not contain two-way assignments.
    These are instead generated here, ensuring that the symmetry is always two-way.

    File format:
    symbol_key_from -- symmetry1 symmetry2 ... symmetryn --> symbol_key_to

    :return: A dictionary mapping symbol keys to tuples of target symbol key and list of symmetries.
    """
    with resource_path(symbol_metadata, "symmetry_other.txt") as file_path:
        with open(file_path, "r") as file:
            lines = file.readlines()

    pattern = re.compile(r"(\S+) -- (.*?) ?--> (\S+)")
    symmetries = defaultdict(list)
    for line in lines:
        match = pattern.match(line)
        if not match:
            logger.error(f"Failed to parse line: {line.strip()}")
            continue
        key_from = match.group(1)
        key_to = match.group(3)
        symmetry_list = [st.Symmetry(sym) for sym in match.group(2).split()]
        inverted_symmetry_list = [sym.invert() for sym in symmetry_list]
        symmetries[key_from].append((key_to, inverted_symmetry_list))
        symmetries[key_to].append((key_from, symmetry_list))
    return symmetries


def sys_virtual_memory_total() -> int:
    """
    Get the total amount of RAM in the system in bytes.
    """
    try:
        return psutil.virtual_memory().total
    except Exception as e:
        logger.exception(e)
        return 0


def sys_swap_memory_total() -> int:
    """
    Get the total amount of swap memory in the system in bytes.
    """
    try:
        return psutil.swap_memory().total
    except Exception as e:
        logger.exception(e)
        return 0


def get_model_path():
    with resources.path(handtex.data.model, "handtex.safetensors") as model_path:
        return model_path


def get_encodings_path():
    with resources.path(handtex.data.model, "encodings.txt") as encodings_path:
        return encodings_path
