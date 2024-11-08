import yaml
import hashlib
import json
from attrs import define, field

from typing import Optional


@define
class Symbol:
    command: str
    package: Optional[str] = None
    fontenc: Optional[str] = None
    mathmode: bool = False
    textmode: bool = True
    key: str = field(init=False)
    filename: str = field(init=False)

    def __post_init__(self):
        # Create an ID for the symbol
        package = self.package or "latex2e"
        fontenc = self.fontenc or "OT1"
        self.key = f"{package}-{fontenc}-{self.command.replace('\\', '_')}"
        # Generate initial filename based on hash of the ID
        self.filename = hashlib.md5(self.key.encode()).hexdigest().upper()

    def __getitem__(self, key):
        if key in self.__annotations__ or key == "key":
            return getattr(self, key)
        raise KeyError(f"Invalid key: {key}")

    def __str__(self):
        return f"{self.command} ({self.package or 'latex2e'}, {self.fontenc or 'OT1'})"

    def to_hash(self):
        result = {
            attr: getattr(self, attr)
            for attr in self.__annotations__
            if getattr(self, attr) is not None
        }
        result["key"] = self.key
        return result

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, "r") as f:
            symbols = yaml.safe_load(f)

        symbol_list = []
        for symbol in symbols:
            if isinstance(symbol, str):
                symbol_list.append(cls(command=symbol))
            elif isinstance(symbol, dict):
                mode_mappings = {
                    "textmode": {"textmode": True, "mathmode": False},
                    "mathmode": {"textmode": False, "mathmode": True},
                    "bothmodes": {"textmode": True, "mathmode": True},
                }
                for mode, mode_args in mode_mappings.items():
                    if mode in symbol:
                        for cmd in symbol[mode]:
                            symbol_list.append(
                                cls(
                                    command=cmd,
                                    package=symbol.get("package"),
                                    fontenc=symbol.get("fontenc"),
                                    **mode_args,
                                )
                            )

        return symbol_list

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            symbols = json.load(f)

        return [cls(**symbol) for symbol in symbols]

    @staticmethod
    def to_json(symbols, json_file):
        with open(json_file, "w") as f:
            json.dump([symbol.to_hash() for symbol in symbols], f, indent=2)
