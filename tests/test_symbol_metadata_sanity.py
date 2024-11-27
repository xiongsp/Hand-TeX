import re
from importlib import resources
from pathlib import Path

import handtex.utils as ut
import structures
from handtex.data import symbol_metadata
import handtex.structures as st


def test_similar_lists_disjunct() -> None:
    """
    Test that the symbol lists are disjunct.
    """
    with resources.path(symbol_metadata, "") as metadata_dir:
        metadata_dir = Path(metadata_dir)
    files = list(metadata_dir.glob("similar*"))

    # Check that each file is consistent on it's own.
    for file in files:
        line_symbols: list[set[str]]
        line_symbols = []

        with file.open("r") as f:
            lines = f.readlines()
        for line in lines:
            symbols = line.strip().split()
            line_symbol_set = set()
            for s in symbols:
                assert (
                    s not in line_symbol_set
                ), f"Symbol {s} is listed twice in {file}, line {line}"
                line_symbol_set.add(s)
            assert line_symbol_set not in line_symbols, f"Line {line} is listed twice in {file}"
            line_symbols.append(line_symbol_set)

        # Check that no two lines are a subset of each other.
        for i, line1 in enumerate(line_symbols, 1):
            for j, line2 in enumerate(line_symbols, 1):
                if i == j:
                    continue
                assert not line1.issubset(line2), f"Line {i} is a subset of line {j} in {file}"

        # Check that all lines are disjunct.
        for i, line1 in enumerate(line_symbols, 1):
            for j, line2 in enumerate(line_symbols, 1):
                if i == j:
                    continue
                assert not line1 & line2, f"Line {i} and line {j} share symbols in {file}"

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


def test_symbol_name_collisions() -> None:
    """
    Look for symbols that have the same command, but aren't
    in a similarity relation.
    """
    # Load symbols.
    symbols = ut.load_symbols()
    # Load similarity relations.
    similarity = ut.load_symbol_metadata_similarity()

    # Check for collisions.
    for symbol1 in symbols.values():
        for symbol2 in symbols.values():
            if symbol1 == symbol2:
                continue
            if symbol1.command == symbol2.command:
                # Check if they are in a similarity relation.
                if symbol1.key in similarity[symbol2.key]:
                    continue
                if symbol2.key in similarity[symbol1.key]:
                    continue
                raise AssertionError(
                    f"Symbols {symbol1.key} and {symbol2.key} have the same command."
                )


def test_self_symmetry_similarity_conflict() -> None:
    """
    Test that only the leader of a similarity group has self-symmetry.
    """

    similarity: dict[str, tuple[str, ...]] = ut.load_symbol_metadata_similarity()
    symbols: dict[str, st.Symbol] = ut.load_symbols()
    leaders: list[str] = ut.select_leader_symbols(list(symbols.keys()), similarity)

    symmetries: dict[str, list[st.Transformation]] = ut.load_symbol_metadata_self_symmetry()

    for leader in leaders:
        if leader not in symmetries:
            # No similars get to have self-symmetry.
            for similar in similarity.get(leader, []):
                assert similar not in symmetries, f"Similar {similar} has self-symmetry."
        else:
            # Only the leader gets to have self-symmetry.
            # Or at least they need to have identical self-symmetry.
            for similar in similarity.get(leader, []):
                if similar not in symmetries:
                    continue
                assert set(symmetries.get(leader, [])) == set(
                    symmetries.get(similar, [])
                ), f"Similar {similar} has different self-symmetry from leader {leader}."
