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
                    os.remove(os.path.join(symbols_dir, duplicate))
        else:
            # If there's only one file, it maps to itself
            filename_mapping[filenames[0]] = filenames[0]

    # Update symbols_data to relink duplicates to the representative filename
    for symbol in symbols_data:
        original_filename = symbol["filename"] + ".svg"
        if original_filename in filename_mapping:
            new_filename = os.path.splitext(filename_mapping[original_filename])[0]
            symbol["filename"] = new_filename

    # Overwrite symbols.json with the updated data
    with open(symbols_json, "w") as f:
        json.dump(symbols_data, f, indent=2)


if __name__ == "__main__":
    main()
