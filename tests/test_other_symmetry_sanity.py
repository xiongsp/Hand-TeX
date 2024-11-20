import re
from importlib import resources
from pathlib import Path

import handtex.utils as ut
from data import symbol_metadata
from handtex.data import symbol_metadata

"""
Example:

amssymb-OT1-_Cap -- rot180 mir0 --> amssymb-OT1-_Cup
amssymb-OT1-_Cap -- rot90 mir135 --> amssymb-OT1-_Subset
"""


def test_duplicates() -> None:
    """
    Test that there are no duplicate lines.
    """
    with ut.resource_path(symbol_metadata, "symmetry_other.txt") as file:
        with file.open("r") as f:
            lines = f.readlines()
    lines_seen = []
    for index, line in enumerate(lines, 1):
        assert line not in lines_seen, f"Line {index}: {line.strip()} is a duplicate."


def test_two_way_assignment() -> None:
    """
    Test that there are no two-way assignments.
    Each assignment is implicitly two-way on it's own.
    """
    with ut.resource_path(symbol_metadata, "symmetry_other.txt") as file:
        with file.open("r") as f:
            lines = f.readlines()

    pattern = re.compile(r"(\S+) -- (.*?) ?--> (\S+)")
    pairs_seen = set()
    for index, line in enumerate(lines, 1):
        match = pattern.match(line)
        assert match, f"Line {index}: {line.strip()} does not match the pattern."
        pair = tuple(sorted((match.group(1), match.group(3))))
        assert pair not in pairs_seen, f"Line {index}: {line.strip()} is a two-way assignment."
        pairs_seen.add(pair)
