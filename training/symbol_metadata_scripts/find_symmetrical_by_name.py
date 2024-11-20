import re
from itertools import chain
from tqdm import tqdm


def parse_symbols(file_path):
    # Step 1: Extract symbol names from file
    symbols = []
    with open(file_path, "r") as file:
        for line in file:
            symbol = line.split(",")[0]
            symbols.append(symbol)
    return symbols


def find_flipped_pairs(symbols, opposing_pairs):
    # Step 2: Precompile regex patterns for opposing pairs
    compiled_patterns = [(re.compile(re.escape(a)), b) for a, b in opposing_pairs]

    # Step 3: Find matching pairs based on opposing substrings
    flipped_pairs = []
    for i, symbol in enumerate(tqdm(symbols)):
        for j, candidate in enumerate(symbols):
            if i != j:  # Avoid comparing the symbol with itself
                for pattern, replacement in compiled_patterns:
                    if pattern.search(symbol):
                        flipped_version = pattern.sub(replacement, symbol)
                        if flipped_version == candidate:
                            flipped_pairs.append((symbol, candidate))
    return flipped_pairs


# Define opposing word pairs to target
def generate_opposing_word_pairs(pairs):
    # Automatically generate flipped versions of tuples
    flipped_added = list(chain.from_iterable([(a, b), (b, a)] for a, b in pairs))
    # Also add a variant with the first letter capitalized, as well as all caps.
    final = [(a.capitalize(), b.capitalize()) for a, b in flipped_added]
    final.extend([(a.upper(), b.upper()) for a, b in flipped_added])
    return flipped_added + final


# Updated base word pairs inspired by symbol list
base_word_pairs = [
    # ("right", "left"),  # Added lowercase handling for Right/Left
    # ("up", "down"),  # Added lowercase handling for Up/Down
    # ("greater", "less"),  # Added lowercase handling for Greater/Less
    # ("in", "out"),  # Added lowercase handling for In/Out
    # ("r", "l"),  # Added lowercase handling for R/L
    # ("u", "d"),  # Added lowercase handling for U/D
    # ("leftright", "rightleft"),
    # ("subset", "supset"),
    # ("cap", "cup"),
    # ("leq", "geq"),
    # ("prec", "succ"),
    # ("langle", "rangle"),
    # ("lfloor", "rfloor"),
    # ("ll", "gg"),
    # ("lt", "gt"),
    ("rising", "falling"),
    ("less", "gtr"),
]

opposing_word_pairs = generate_opposing_word_pairs(base_word_pairs)

# Parse the symbols from the given file
symbols_list = parse_symbols("../../handtex/data/symbol_metadata/symbol_frequency.csv")

# Find and print the matching pairs
matching_pairs = find_flipped_pairs(symbols_list, opposing_word_pairs)
# For each pair, sort them alphabetically within the pair.
matching_pairs = [tuple(sorted(pair)) for pair in matching_pairs]
# Remove duplicates by converting to a set and back to a list
matching_pairs = list(set(matching_pairs))
# Sort the pairs alphabetically by the first element
matching_pairs.sort(key=lambda pair: pair[0])
# Remove any that contain a {
matching_pairs = [pair for pair in matching_pairs if "{" not in pair[0] and "{" not in pair[1]]
for pair in matching_pairs:
    print(f"{pair[0]} -- -> {pair[1]}")
