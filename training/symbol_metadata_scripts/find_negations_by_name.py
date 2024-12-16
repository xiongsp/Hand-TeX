from tqdm import tqdm

import handtex.symbol_relations as sr


def parse_symbols(file_path):
    # Step 1: Extract symbol names from file
    symbols = []
    with open(file_path, "r") as file:
        for line in file:
            symbol = line.split(",")[0]
            symbols.append(symbol)
    return symbols


def get_command(symbol):
    command = symbol.split("-", maxsplit=2)[2]
    if command.startswith("_"):
        command = command[1:]
    return command


def is_negation_of(s1, s2):
    """
    Check if s1 is a negation of s2.
    """
    # Step 2: Check if one symbol is a negation of the other
    if s1 == s2:
        return False
    c1 = get_command(s1)
    c2 = get_command(s2)
    if c1 == c2:
        return False
    if c1 == "n" + c2:
        return True
    if c1 == "N" + c2:
        return True
    if c1 == "not" + c2:
        return True
    if c1 == "not_" + c2:
        return True
    if c1 == "Not" + c2:
        return True
    if c1 == "Not_" + c2:
        return True
    return False


def find_negated_pairs(symbols):
    negated_pairs = []
    # O(not so sodding terrible)
    for symbol in tqdm(symbols):
        for candidate in symbols:
            if is_negation_of(symbol, candidate):
                negated_pairs.append((symbol, candidate))
    return negated_pairs


symbol_data = sr.SymbolData()

# Parse the symbols from the given file
symbols_list = symbol_data.leaders

# Find and print the matching pairs
matching_pairs = find_negated_pairs(symbols_list)
# For each pair, sort them alphabetically within the pair.
matching_pairs = list(set(matching_pairs))
# Sort the pairs alphabetically by the first element
matching_pairs.sort(key=lambda pair: pair[0])
matching_pairs.sort(key=lambda pair: pair[0].startswith("latex2e"), reverse=True)

for s_negated, s_normal in matching_pairs:
    if s_normal in symbol_data.get_symmetry_group(
        s_negated
    ) or s_negated in symbol_data.get_symmetry_group(s_normal):
        continue
    print(f"{s_normal} -/ /- {s_negated}")
