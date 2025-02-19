import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class Symbol:
    command: str
    package: Optional[str] = None
    fontenc: Optional[str] = None
    mathmode: bool = False
    textmode: bool = True
    pdflatex: bool = True
    key: str = field(init=False)
    filename: str = field(init=False)

    def __post_init__(self):
        # Create an ID for the symbol
        package = self.package or "latex2e"
        fontenc = self.fontenc or "OT1"
        self.key = f"{package}-{self.command.replace('\\', '_')}"
        # Generate initial filename based on hash of the ID
        self.filename = hashlib.md5(self.key.encode()).hexdigest().upper()

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
                                    pdflatex=symbol.get("pdflatex", True),
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


def main():

    # Load symbols from the YAML file
    symbols = Symbol.from_yaml("symbols.yaml")

    # Check the list for duplicates: two symbols with the same key.
    seen = set()
    for symbol in symbols:
        if symbol.key in seen:
            print(f"Duplicate symbol: {symbol.key}")
        seen.add(symbol.key)

    # Post-processing step: shorten filenames without causing collisions and ensure all hashes have the same length
    # Perform a binary search to find the minimum hash length that avoids collisions
    def has_collision(candidate_length):
        seen = set()
        for symbol in symbols:
            short_hash = symbol.filename[:candidate_length]
            if short_hash in seen:
                return True
            seen.add(short_hash)
        return False

    min_length = 1
    max_length = len(symbols[0].filename) if symbols else 0

    while min_length < max_length:
        mid_length = (min_length + max_length) // 2
        if has_collision(mid_length):
            min_length = mid_length + 1
        else:
            max_length = mid_length

    # Ensure all filenames have the same length (minimum non-colliding length)
    final_length = min_length
    for symbol in symbols:
        symbol.filename = symbol.filename[:final_length]

    import handtex.utils as ut
    import handtex.data.symbol_metadata

    # Might need to re-run the script a few times in case one of the inkscapes gets killed.
    # Only an issue if multi-threading. But you really should multithread with 1000 symbols...
    symbol_path = ut.resource_path(handtex.data.symbol_metadata) / "symbols.json"

    # Save the symbols to a JSON file
    Symbol.to_json(symbols, symbol_path)


if __name__ == "__main__":
    main()
