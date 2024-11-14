import handtex.utils as ut
import re
import difflib
import os
import platform
import shutil
import sys
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


def test_identical_lists_disjunct() -> None:
    """
    Test that the identical symbol lists are disjunct.
    """
    with resources.path(symbol_metadata, "") as metadata_dir:
        metadata_dir = Path(metadata_dir)
    files = list(metadata_dir.glob("identical*"))

    # There cannot be the same key listed twice in the file.
    for file in files:
        # Just parse out all the keys with a regex.
        # A key is a string of 1 or more characters, no whitespace.
        keys = list(re.findall(r"\S+", file.read_text()))
        key_set = set(keys)
        for key in key_set:
            keys.remove(key)
        assert not keys, f"File {file} contains duplicate keys: {keys}"


def test_similar_lists_disjunct() -> None:
    """
    Test that the identical symbol lists are disjunct.
    """
    with resources.path(symbol_metadata, "") as metadata_dir:
        metadata_dir = Path(metadata_dir)
    files = list(metadata_dir.glob("similar*"))

    # There cannot be the same key listed twice in the file.
    global_keys = set()
    for file in files:
        # Just parse out all the keys with a regex.
        # A key is a string of 1 or more characters, no whitespace.
        keys = list(re.findall(r"\S+", file.read_text()))
        key_set = set(keys)
        for key in key_set:
            keys.remove(key)
        assert not keys, f"File {file} contains duplicate keys: {keys}"

        # The keys in this file must not be in the global set.
        intersection = global_keys & key_set
        assert (
            not intersection
        ), f"File {file} contains keys already in another file: {intersection}"
        global_keys |= key_set
