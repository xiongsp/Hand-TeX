import json
import shutil
from pathlib import Path
from typing import Callable, Any

import attrs
from attrs import define, Factory
from loguru import logger

import handtex.utils as ut


@define
class Config:
    gui_theme: str = ""  # Blank means system theme.
    stroke_width: int = 6
    new_data_dir: str = "new_data"
    scroll_on_draw: bool = True
    single_click_to_copy: bool = True
    disabled_packages: list[str] = Factory(list)

    def save(self, path: Path = None) -> bool:
        """
        Write to a temporary file and then move it to the destination.
        If the write fails, the temporary file is deleted.
        When the path is None, the config is saved to the default location.

        :param path: [Optional] The path to write the profile to.
        :return: True if the profile was written successfully, False otherwise.
        """
        logger.debug(f"Saving config")

        if path is None:
            path = ut.get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        success = self._unsafe_save(temp_path)
        if success:
            try:
                shutil.move(temp_path, path)
            except Exception:
                logger.exception(f"Failed to rename {temp_path} to {path}")
                success = False
        if not success:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                logger.exception(f"Failed to delete {temp_path}")
        return success

    def _unsafe_save(self, path: Path) -> bool:
        """
        Write the config to a file.

        :param path: The path to write the config to.
        :return: True if the config was written successfully, False otherwise.
        """
        logger.debug("Writing config to disk...")
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(self.dump(), file, indent=4)
            return True
        except Exception:
            logger.exception(f"Failed to write profile to {path}")
            return False

    def dump(self) -> dict:
        """
        Dump the config to a dictionary.
        """
        return attrs.asdict(self)

    def pretty_log(self) -> None:
        """
        Log the config in a safe, easily readable format to debug.
        """
        data = self.dump()
        logger.debug("Config:\n" + json.dumps(data, indent=4))


def load_config(conf_path: Path, config_factory: Callable[[], Any] = Config) -> tuple[
    Config,
    list[ut.RecoverableParseException],
    list[ut.CriticalParseError],
]:
    """
    Load the configuration from the given path.
    If any errors occur, they will be returned as Exception objects in the corresponding
    severity list.

    :param conf_path: Path to the configuration file.
    :param config_factory: [Optional] Factory function to create a new config object.
    :return: The configuration and 2 lists of errors.
    """

    config = config_factory()

    recoverable_exceptions: list[ut.RecoverableParseException] = []
    critical_errors: list[ut.CriticalParseError] = []

    try:
        with open(conf_path, "r") as f:
            # Load the default config, then update it with the values from the config file.
            # This way, missing values will be filled in with the default values.
            json_data = json.load(f)

            recoverable_exceptions = ut.load_dict_to_attrs_safely(config, json_data)

    except OSError as e:
        logger.exception(f"Failed to read config file {conf_path}")
        critical_errors = [
            ut.CriticalParseError(
                f"{type(e).__name__}: Failed to read config file {conf_path}: {e}"
            )
        ]
    except json.decoder.JSONDecodeError as e:
        logger.exception(f"Configuration file could not be parsed {conf_path}")
        critical_errors = [
            ut.CriticalParseError(
                f"{type(e).__name__}: Configuration file could not be parsed {conf_path}: {e}"
            )
        ]

    # If's fubar, reset the config just to be sure.
    if critical_errors:
        config = config_factory()

    return config, recoverable_exceptions, critical_errors
