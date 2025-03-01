import sqlite3
import json
from collections import defaultdict

from tqdm import tqdm

import handtex.utils as ut
import handtex.detector.inference as inf
import training.train as trn
import handtex.detector.image_gen as ig
import handtex.symbol_relations as sr


# Defunct tool to use the model to find symmetrical symbols automatically.


def main2():
    """
    Grab all the new symbol keys.
    """
    symbol_data = sr.SymbolData()

    since_key = "stix2-_fullouterjoin"
    all_keys = symbol_data.all_keys
    start_index = all_keys.index(since_key)
    keys = all_keys[start_index:]

    # Also remove all symbols that aren't leaders.
    keys = filter(lambda key: key in symbol_data.leaders, keys)

    for key in keys:
        print(f"{key}: ")


def main():
    """
    Check for symbols that have axes of symmetry or rotational symmetry.
    """

    db_path = "../database/handtex.db"

    """
    CREATE TABLE samples (
        id INTEGER PRIMARY KEY,
        key TEXT,
        strokes TEXT
    )
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    model, label_decoder = inf.load_model_and_decoder(
        ut.get_model_path(), trn.num_classes, ut.get_encodings_path()
    )

    symbols = ut.load_symbols()
    similar_symbols = ut.load_symbol_metadata_similarity()
    leaders: list[str] = ut.select_leader_symbols(list(symbols.keys()), similar_symbols)

    min_confidence = 0.95
    min_overlap = 1

    # Track the symbols that overlapped with each other.
    collisions: dict[str, dict[str, list[tuple[str, float]]]] = {}
    collision_types = (
        # ("rot45", ig.rotation_matrix(45)),
        ("rot90", ig.rotation_matrix(90)),
        # ("rot135", ig.rotation_matrix(135)),
        ("rot180", ig.rotation_matrix(180)),
        # ("rot225", ig.rotation_matrix(225)),
        ("rot270", ig.rotation_matrix(270)),
        # ("rot315", ig.rotation_matrix(315)),
        ("sym_0", ig.reflection_matrix(0)),
        # ("sym_22.5", ig.reflection_matrix(22.5)),
        ("sym_45", ig.reflection_matrix(45)),
        # ("sym_67.5", ig.reflection_matrix(67.5)),
        ("sym_90", ig.reflection_matrix(90)),
        # ("sym_112.5", ig.reflection_matrix(112.5)),
        ("sym_135", ig.reflection_matrix(135)),
        # ("sym_157.5", ig.reflection_matrix(157.5)),
    )
    for collision_type, _ in collision_types:
        collisions[collision_type] = defaultdict(list)

    for leader in tqdm(leaders):
        # Load random samples per leader and check the predictions.
        max_samples = 9
        cursor.execute(
            "SELECT key, strokes FROM samples WHERE key = ? LIMIT ?", (leader, max_samples)
        )
        samples = cursor.fetchall()

        current_collisions: dict[str, dict[str, list[tuple[str, float]]]] = {}
        for col_type, _ in collision_types:
            current_collisions[col_type] = defaultdict(list)

        for sample in samples:
            key, strokes = sample
            strokes = json.loads(strokes)
            for col_type, matrix in collision_types:
                strokes = ig.apply_transformations(strokes, matrix)
                predictions = inf.predict_strokes(strokes, model, label_decoder)

                for symbol, confidence in predictions:
                    if confidence < min_confidence:
                        continue

                    current_collisions[col_type][symbol].append((key, confidence))
            # Check if there was sufficient overlap for each key found in the collisions.
            for col_type, _ in collision_types:
                for symbol, keys in current_collisions[col_type].items():
                    if len(keys) / len(samples) >= min_overlap:
                        # Add only a single entry to the collisions dictionary,
                        # using the median confidence.
                        confidences = [conf for _, conf in keys]
                        median_conf = sorted(confidences)[len(confidences) // 2]
                        collisions[col_type][symbol].append((leader, median_conf))

    conn.close()

    print("Results:")
    for col_type, _ in collision_types:
        print(f"Collision Type: {col_type}")
        for symbol, collision_list in collisions[col_type].items():
            print(f"Symbol: {symbol}")
            for leader, confidence in collision_list:
                print(f"Leader: {leader}, Confidence: {confidence}")


def evaluate_results():

    # collision_types = (
    #     ("rot45", ig.rotation_matrix(45)),
    #     ("rot90", ig.rotation_matrix(90)),
    #     ("rot135", ig.rotation_matrix(135)),
    #     ("rot180", ig.rotation_matrix(180)),
    #     ("rot225", ig.rotation_matrix(225)),
    #     ("rot270", ig.rotation_matrix(270)),
    #     ("rot315", ig.rotation_matrix(315)),
    #     ("sym_0", ig.reflection_matrix(0)),
    #     ("sym_22.5", ig.reflection_matrix(22.5)),
    #     ("sym_45", ig.reflection_matrix(45)),
    #     ("sym_67.5", ig.reflection_matrix(67.5)),
    #     ("sym_90", ig.reflection_matrix(90)),
    #     ("sym_112.5", ig.reflection_matrix(112.5)),
    #     ("sym_135", ig.reflection_matrix(135)),
    #     ("sym_157.5", ig.reflection_matrix(157.5)),
    # )
    # example:
    """
    Collision Type: rot90
    Symbol: wasysym-_sun
    Leader: wasysym-_sun, Confidence: 0.9998611211776733
    Symbol: wasysym-_wasylozenge
    Leader: wasysym-_wasylozenge, Confidence: 0.9990512728691101
    Symbol: marvosym-_Bat
    Leader: marvosym-_Bat, Confidence: 0.9999992847442627
    Collision Type: rot45
    Symbol: marvosym-_Mundus
    Leader: marvosym-_Mundus, Confidence: 0.9997351765632629
    Symbol: marvosym-_Yinyang
    Leader: marvosym-_Yinyang, Confidence: 0.9999496936798096
    Symbol: marvosym-_MVAt
    Leader: marvosym-_MVAt, Confidence: 0.9999268054962158
    Symbol: stix2-_fullouterjoin
    Leader: stix2-_fullouterjoin, Confidence: 0.9992081522941589
    """
    path = "../misc_scripts/scratch.txt"
    with open(path, "r") as file:
        lines = file.readlines()

    collisions: dict[str, dict[str, list[str]]] = {}
    col_type = ""
    symbol = ""
    for line in lines:
        parts = line.strip().split()
        if line.startswith("Collision"):
            col_type = parts[-1]
            collisions[col_type] = defaultdict(list)
        elif line.startswith("Symbol"):
            symbol = parts[-1]
            collisions[col_type][symbol] = []
        elif line.startswith("Leader"):
            leader = parts[1].replace(",", "")
            collisions[col_type][symbol].append(leader)

    # Perform analysis on the collisions.
    # dict[symbol, dict[col_type, list[leader]]]
    collisions_by_symbol: dict[str, dict[str, list[str]]] = {}
    symbols = set()
    for category_dict in collisions.values():
        for symbol, collision_list in category_dict.items():
            symbols.add(symbol)
    for symbol in symbols:
        collisions_by_symbol[symbol] = {}
        for col_type in collisions.keys():
            collisions_by_symbol[symbol][col_type] = []

    for col_type, category_dict in collisions.items():
        for symbol, collision_list in category_dict.items():
            collisions_by_symbol[symbol][col_type].extend(collision_list)

    # Find self-similar collisions.
    # dict[symbol, list[col_type]]
    self_similar_collisions: dict[str, list[str]] = defaultdict(list)
    for symbol, category_dict in collisions_by_symbol.items():
        for category, col_list in category_dict.items():
            if symbol in col_list:
                self_similar_collisions[symbol].append(category)

    # Remove self-similar collisions from collisions_by_symbol.
    # for symbol, col_dict in collisions_by_symbol.items():
    for symbol in symbols:
        col_dict = collisions_by_symbol[symbol]
        for col_type in col_dict:
            col_dict[col_type] = [leader for leader in col_dict[col_type] if leader != symbol]
        if not any(col_dict.values()):
            del collisions_by_symbol[symbol]

    # All transformations:
    circle_sym = {
        "rot45",
        "rot90",
        "rot135",
        "rot180",
        "rot225",
        "rot270",
        "rot315",
        "sym_0",
        "sym_22.5",
        "sym_45",
        "sym_67.5",
        "sym_90",
        "sym_112.5",
        "sym_135",
        "sym_157.5",
    }
    square_sym = {"rot90", "rot180", "rot270", "sym_0", "sym_45", "sym_90", "sym_135"}
    diamond_sym = {
        "rot45",
        "rot135",
        "rot225",
        "rot315",
        "sym_22.5",
        "sym_67.5",
        "sym_112.5",
        "sym_157.5",
    }
    diag_45_sym = {"rot180", "sym_45"}
    diag_135_sym = {"rot180", "sym_135"}
    mirror_vert_sym = {"rot180", "sym_0"}
    mirror_hor_sym = {"rot180", "sym_90"}

    symmetry_types = {
        "circle_sym": circle_sym,
        "square_sym": square_sym,
        "diamond_sym": diamond_sym,
        "diag_45_sym": diag_45_sym,
        "diag_135_sym": diag_135_sym,
        "mirror_vert_sym": mirror_vert_sym,
        "mirror_hor_sym": mirror_hor_sym,
    }

    # Analyze the results.
    # Find these specific categories of symmetry.

    def check_group(c_list: list[str]) -> str | None:
        # Find the largest collision group this matches.
        for group, group_set in symmetry_types.items():
            if group_set.issubset(set(c_list)):
                return group
        return None

    # Check the self-similar collisions first.
    print("Self-similar collisions:")
    for symbol, col_list in self_similar_collisions.items():
        group = check_group(col_list)
        if group is not None:
            print(f"{symbol} | {group}")
        else:
            print(f"{symbol} | {', '.join(col_list)}")

    # Check the other collisions.
    print("Other collisions:")
    for symbol, col_dict in collisions_by_symbol.items():
        for col_type, col_list in col_dict.items():
            if not col_list:
                continue
            group = check_group(col_list)
            if group is not None:
                print(f"{symbol}: Homomorphisms: {', '.join(col_list)} | {group}")
            else:
                print(f"{symbol}: Homomorphisms: {', '.join(col_list)} | {col_type}")


if __name__ == "__main__":
    # main()
    # evaluate_results()
    main2()
