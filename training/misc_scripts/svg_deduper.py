import os
import json
import hashlib
from collections import defaultdict


# Function to calculate md5 hash of a file
def calculate_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def main():
    symbols_json = "../../handtex/data/symbol_metadata/symbols.json"

    # Load symbols data from symbols.json
    with open(symbols_json, "r") as f:
        symbols_data = json.load(f)

    # Create a mapping from filename key to command
    filename_to_key = {symbol["filename"]: symbol for symbol in symbols_data}

    # Assert no key has a space in it.
    for key in filename_to_key.values():
        assert " " not in key["key"], f"Key '{key['key']}' contains a space."

    # Directory containing SVG files
    symbols_dir = "../../handtex/data/symbols"

    # Dictionary to store md5 hashes and corresponding filenames
    hash_to_filenames = defaultdict(list)

    # Traverse the symbols directory and calculate md5 hashes
    for filename in os.listdir(symbols_dir):
        if filename.endswith(".svg"):
            filepath = os.path.join(symbols_dir, filename)
            file_hash = calculate_md5(filepath)
            hash_to_filenames[file_hash].append(filename)

    # Disable if you just want to see the duplicates without deleting them.
    do_cleanup = True

    # Prepare to purge duplicates and select a single representative for each duplicate group
    filename_mapping = {}
    for filenames in hash_to_filenames.values():
        if len(filenames) > 1:
            # Choose the first filename as the representative
            representative = filenames[0]
            for duplicate in filenames:
                filename_mapping[duplicate] = representative
                if duplicate != representative:
                    # Remove the duplicate file
                    if do_cleanup:
                        os.remove(os.path.join(symbols_dir, duplicate))
        else:
            # If there's only one file, it maps to itself
            filename_mapping[filenames[0]] = filenames[0]

    if do_cleanup:
        # Update symbols_data to relink duplicates to the representative filename
        for symbol in symbols_data:
            original_filename = symbol["filename"] + ".svg"
            if original_filename in filename_mapping:
                new_filename = os.path.splitext(filename_mapping[original_filename])[0]
                symbol["filename"] = new_filename

        # Overwrite symbols.json with the updated data
        with open(symbols_json, "w") as f:
            json.dump(symbols_data, f, indent=2)

    import handtex.symbol_relations as sr

    symbol_data_obj = sr.SymbolData()

    # Collect the groups of symbol keys with identical hashes.
    identical_groups = [group for group in hash_to_filenames.values() if len(group) > 1]
    # Convert the file names to symbol keys.
    identical_groups = [
        [filename_to_key[filename[:-4]]["key"] for filename in group] for group in identical_groups
    ]
    # Check if any of the symbols in each group have a similarity group larger than 1.
    # If so, check if the other members of the group are also in the group.
    # If they are not, propose adding the missing ones.
    for group in identical_groups:
        if all(len(symbol_data_obj.get_similarity_group(symbol)) == 1 for symbol in group):
            print(f"+ Make a new group for \n{group}\n")
        else:
            for symbol in group:
                similarity_group = symbol_data_obj.get_similarity_group(symbol)
                if len(similarity_group) > 1:
                    leftovers = set(group) - set(similarity_group)
                    if leftovers:
                        print(f"Add {leftovers} to {similarity_group}\n")


if __name__ == "__main__":
    main()
