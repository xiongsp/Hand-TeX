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


class Transformation(StrEnum):
    identity = auto()
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

    def is_identity(self):
        return self.name == "identity"

    def is_rotation(self):
        return self.name.startswith("rot")

    def is_mirroring(self):
        return self.name.startswith("mir")

    @property
    def angle(self):
        return int(self.name[3:])

    def invert(self) -> "Transformation":
        if self.is_rotation():
            return Transformation(f"rot{360 - self.angle}")
        else:
            return self

    # Type if python typing didn't suck:
    # def merge(self, other: "Transformation") -> "Transformation" | list["Transformation"]:
    def merge(self, other):
        if self.is_identity():
            return other
        if other.is_identity():
            return self
        if self.is_rotation() and other.is_rotation():
            if (self.angle + other.angle) % 360 == 0:
                return Transformation.identity
            return Transformation(f"rot{(self.angle + other.angle) % 360}")
        if self.is_mirroring() and other.is_mirroring():
            if self.angle == other.angle:
                return Transformation.identity
            return [self, other]
        return [self, other]

    def can_merge(self, other):
        if self.is_identity() or other.is_identity():
            return True
        if self.is_rotation() and other.is_rotation():
            return True
        if self.is_mirroring() and other.is_mirroring():
            if self.angle == other.angle:
                return True
        return False


def simplify_transformations(
    transforms: tuple[tuple[Transformation, ...], ...]
) -> tuple[tuple[Transformation, ...], ...]:
    """
    Simplify a list of transformations by merging compatible transformations.
    Identity transformations are removed.
    """
    match transforms:
        case ((),), ((Transformation.identity,),):
            return ((),)
        case ((t,),) if isinstance(t, Transformation):
            return ((t,),)

    simplified = []
    for transform in transforms:
        simplified_transform = []
        for t in transform:
            if simplified_transform:
                last = simplified_transform[-1]
                if last.can_merge(t):
                    simplified_transform[-1] = last.merge(t)
                    continue
            simplified_transform.append(t)
        # If we just ended up with identity, remove it.
        if simplified_transform != [Transformation.identity]:
            simplified.append(tuple(simplified_transform))
        else:
            simplified.append(())
    # Ensure there are no duplicates.
    simplified = tuple(set(simplified))
    return simplified
