import json
from enum import Enum, auto
from math import sin, cos, pi
from typing import overload

from attrs import frozen


@frozen
class Symbol:
    command: str
    package: str
    fontenc: str
    mathmode: bool
    textmode: bool
    pdflatex: bool
    key: str
    filename: str

    def __str__(self):
        return self.key

    @classmethod
    def from_dict(cls, symbol) -> "Symbol":
        if "package" not in symbol:
            symbol["package"] = "latex2e"
        if symbol["package"] == "logix":
            symbol["pdflatex"] = False
        else:
            symbol["pdflatex"] = True
        if "fontenc" not in symbol:
            symbol["fontenc"] = "OT1"
        if "key" not in symbol:
            symbol["key"] = symbol["package"] + "-" + symbol["command"].replace("\\", "_")
        if "mode" not in symbol:
            symbol["mathmode"] = True
            symbol["textmode"] = False
        elif symbol["mode"] == "text":
            symbol["mathmode"] = False
            symbol["textmode"] = True
        elif symbol["mode"] == "both":
            symbol["mathmode"] = True
            symbol["textmode"] = True
        symbol.pop("mode", None)
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

    def compiler_str(self) -> str:
        if self.pdflatex:
            return "pdfLaTeX"
        else:
            return "XeLaTeX/LuaLaTeX"

    @classmethod
    def dummy(cls, key: str) -> "Symbol":
        return cls(
            command=key,
            package="?",
            fontenc="?",
            mathmode=False,
            textmode=False,
            pdflatex=True,
            key=key,
            filename="",
        )


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


class Transformation:
    @overload
    def __init__(self, angle: str): ...

    @overload
    def __init__(self, angle: float = 0.0, is_rotation: bool = True): ...

    def __init__(self, angle: float | str = 0.0, is_rotation: bool = True):
        if isinstance(angle, str):
            value = angle
            if value == "identity":
                self.angle = 0.0
                self.is_rotation = True
            elif value.startswith("rot"):
                self.angle = float(value[3:].replace("_", ".")) % 360
                self.is_rotation = True
            elif value.startswith("mir"):
                self.angle = float(value[3:].replace("_", ".")) % 180
                self.is_rotation = False
            else:
                raise ValueError(f"Invalid transformation string: {value}")
        else:
            self.angle = angle % 360 if is_rotation else angle % 180
            self.is_rotation = is_rotation

    def __str__(self):
        if self.is_rotation:
            if self.angle == 0:
                return "identity"
            return f"rot{self.angle}"
        return f"mir{self.angle}"

    def __repr__(self):
        return f"<{self}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transformation):
            return NotImplemented
        return self.angle == other.angle and self.is_rotation == other.is_rotation

    def __hash__(self) -> int:
        return hash((self.angle, self.is_rotation))

    @property
    def is_identity(self):
        return self.is_rotation and self.angle == 0

    def invert(self) -> "Transformation":
        if self.is_rotation:
            return Transformation(angle=(360 - self.angle) % 360, is_rotation=True)
        else:
            return self

    def merge(self, other: "Transformation"):
        if self.is_identity:
            return other
        if other.is_identity:
            return self
        if self.is_rotation and other.is_rotation:
            combined_angle = (self.angle + other.angle) % 360
            if combined_angle == 0:
                return Transformation(angle=0.0, is_rotation=True)
            return Transformation(angle=combined_angle, is_rotation=True)
        if not self.is_rotation and not other.is_rotation:
            if self.angle == other.angle:
                return Transformation(angle=0.0, is_rotation=True)  # Identity
            return [self, other]
        return [self, other]

    def can_merge(self, other: "Transformation") -> bool:
        if self.is_identity or other.is_identity:
            return True
        if self.is_rotation and other.is_rotation:
            return True
        if not self.is_rotation and not other.is_rotation:
            return self.angle == other.angle
        return False

    # Enum-like class attributes for easy access
    @classmethod
    def identity(cls):
        return cls(angle=0.0, is_rotation=True)

    @classmethod
    def rot(cls, angle: float):
        return cls(angle=angle, is_rotation=True)

    @classmethod
    def mir(cls, angle: float):
        return cls(angle=angle, is_rotation=False)


def simplify_transformations(transforms: tuple[Transformation, ...]) -> tuple[Transformation, ...]:
    """
    Simplify a list of transformations by merging compatible transformations.
    Identity transformations are removed.
    """
    match transforms:
        case ():
            return ()
        case (t,) if isinstance(t, Transformation):
            if t.is_identity:
                return ()
            return (t,)

    simplified_transform = []
    for t in transforms:
        if simplified_transform:
            last = simplified_transform[-1]
            if last.can_merge(t):
                simplified_transform[-1] = last.merge(t)
                continue
        simplified_transform.append(t)
    # Trim out identity at the end.
    while simplified_transform and simplified_transform[-1].is_identity:
        simplified_transform.pop()
    # If we just ended up with identity, remove it.
    if simplified_transform != [Transformation.identity()]:
        return tuple(simplified_transform)
    return ()


class Negation:
    """
    Track the position of a straight line to negate a symbol.
    Modifiers:
    - angle: a rotation transformation, starting at 0 degrees being horizontal
    - offset: uangle, oangle, Oangle, Uangle (e.g. u180, o90, O45) to offset the line from the center by 0.15, 0.3, 0.45, 1.1
    - scale: s0.5, s2 to scale the line by 50% or 200% relative to the symbol

    String format examples:
    - "rot45"
    - "rot45 o90 s0.5"
    - "o90"
    - "O45
    - "s2"
    - "slash"  # Preset for a simple slash: rot-22.5

    If the string is empty, assume defaults.
    """

    def __init__(
        self, angle: float, offset_angle: float, offset_factor: float, scale_factor: float
    ):
        self.angle = angle
        self.offset_angle = offset_angle
        self.offset_factor = offset_factor
        self.scale_factor = scale_factor

    @classmethod
    def from_string(cls, negation_str: str) -> "Negation":
        negation_str = negation_str.strip()
        if negation_str == "slash":
            return cls(angle=67.5, offset_angle=0, offset_factor=0, scale_factor=1.5)
        if negation_str == "bar":
            return cls(angle=0, offset_angle=0, offset_factor=0, scale_factor=1)

        angle = 0
        offset_angle = 0
        offset_factor = 0
        scale_factor = 1
        for part in negation_str.split():
            if part.startswith("rot"):
                angle = float(part[3:])
            elif part.startswith("u"):
                offset_angle = float(part[1:])
                offset_factor = 0.15
            elif part.startswith("o"):
                offset_angle = float(part[1:])
                offset_factor = 0.3
            elif part.startswith("O"):
                offset_angle = float(part[1:])
                offset_factor = 0.45
            elif part.startswith("U"):
                offset_angle = float(part[1:])
                offset_factor = 1.2
            elif part.startswith("s"):
                scale_factor = float(part[1:])
        return cls(angle, offset_angle, offset_factor, scale_factor)

    @property
    def vert_angle(self):
        """
        The angle to apply to a vertical line to negate the symbol.
        """
        return self.angle - 90

    @property
    def x_offset(self):
        """
        Calculate the horizontal component of the offset.
        The offset is in terms of the distance from the center to the edge.
        """
        # Convert polar coordinated (in degrees) to cartesian coordinates
        return self.offset_factor * cos(self.offset_angle * pi / 180) / 2

    @property
    def y_offset(self):
        """
        Calculate the vertical component of the offset.
        The offset is in terms of the distance from the center to the edge.
        """
        return self.offset_factor * sin(self.offset_angle * pi / 180) / 2

    def is_slash(self):
        return (
            self.angle == 67.5
            and self.offset_angle == 0
            and self.offset_factor == 0
            and self.scale_factor == 1.5
        )

    def __str__(self):
        if self.is_slash():
            return "slash"
        parts = []
        if self.angle != 0:
            parts.append(f"rot{self.angle}")
        if self.offset_factor != 0:
            if self.offset_factor - 0.15 < 1e-6:
                parts.append(f"u{self.offset_angle}")
            elif self.offset_factor - 0.3 < 1e-6:
                parts.append(f"o{self.offset_angle}")
            elif self.offset_factor - 0.45 < 1e-6:
                parts.append(f"O{self.offset_angle}")
            elif self.offset_factor - 1.2 < 1e-6:
                parts.append(f"U{self.offset_angle}")
            else:
                raise ValueError(f"Invalid offset factor: {self.offset_factor}")
        if self.scale_factor != 1:
            parts.append(f"s{self.scale_factor}")

        if not parts:
            return "bar"
        return " ".join(parts)

    def __repr__(self):
        return f"<Negation({self.angle=}, {self.offset_angle=}, {self.offset_factor=}, {self.scale_factor=})>"


class Inside(Enum):
    Circle = auto()
    Triangle = auto()
    Square = auto()
