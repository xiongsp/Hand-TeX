import json
from enum import StrEnum, auto
from attrs import frozen


@frozen
class Symbol:
    command: str
    package: str
    fontenc: str
    mathmode: bool
    textmode: bool
    key: str
    filename: str

    def __str__(self):
        return self.key

    @classmethod
    def from_dict(cls, symbol) -> "Symbol":
        if "package" not in symbol:
            symbol["package"] = "latex2e"
        if "fontenc" not in symbol:
            symbol["fontenc"] = "OT1"
        return cls(**symbol)

    @classmethod
    def from_json(cls, json_file) -> list["Symbol"]:
        with open(json_file, "r") as f:
            symbols = json.load(f)

        return [cls.from_dict(symbol) for symbol in symbols]

    def package_is_default(self) -> bool:
        return self.package == "latex2e"

    def fontenc_is_default(self) -> bool:
        return self.fontenc == "OT1"

    def mode_str(self) -> str:
        if self.mathmode and not self.textmode:
            return "Mathmode"
        elif self.textmode and not self.mathmode:
            return "Textmode"
        else:
            return "Math & Textmode"


@frozen
class SymbolDrawing:
    key: str
    strokes: list[list[tuple[int, int]]]
    scaling: float
    x_offset: int
    y_offset: int

    def dump(self):
        return {
            "key": self.key,
            "strokes": self.strokes,
        }


class Symmetry(StrEnum):
    none = auto()
    rot45 = auto()
    rot90 = auto()
    rot135 = auto()
    rot180 = auto()
    rot225 = auto()
    rot270 = auto()
    rot315 = auto()
    mir0 = auto()
    mir45 = auto()
    mir90 = auto()
    mir135 = auto()

    def __str__(self):
        return self.name

    def is_none(self):
        return self.name == "none"

    def is_rotation(self):
        return self.name.startswith("rot")

    def is_mirroring(self):
        return self.name.startswith("mir")

    @property
    def angle(self):
        return int(self.name[3:])

    def invert(self) -> "Symmetry":
        if self.is_rotation():
            return Symmetry(f"rot{360 - self.angle}")
        else:
            return self
