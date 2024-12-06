import json

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
        if "fontenc" not in symbol:
            symbol["fontenc"] = "OT1"
        if "pdflatex" not in symbol:
            symbol["pdflatex"] = True
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


from typing import overload, Union


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
    # If we just ended up with identity, remove it.
    if simplified_transform != [Transformation.identity()]:
        return tuple(simplified_transform)
    return ()
