import json
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
    def from_dict(cls, symbol):
        if "package" not in symbol:
            symbol["package"] = "latex2e"
        if "fontenc" not in symbol:
            symbol["fontenc"] = "OT1"
        return cls(**symbol)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            symbols = json.load(f)

        return [cls.from_dict(symbol) for symbol in symbols]


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
