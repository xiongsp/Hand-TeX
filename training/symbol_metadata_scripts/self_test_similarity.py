import sqlite3
import json
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

import handtex.utils as ut
import training.inference as inf
import training.train as trn
import training.image_gen as ig


def main():
    """
    We want to find similar symbols. We will do this by
    performing inference on the current model across all
    leaders and see if any have a high overlap in predictions.
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

    min_confidence = 0.1
    min_overlap = 0.5

    # Track the symbols that overlapped with each other.
    collisions: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for leader in tqdm(leaders):
        # Load random samples per leader and check the predictions.
        max_samples = 9
        cursor.execute(
            "SELECT key, strokes FROM samples WHERE key = ? LIMIT ?", (leader, max_samples)
        )
        samples = cursor.fetchall()

        current_collisions: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for sample in samples:
            key, strokes = sample
            strokes = json.loads(strokes)
            predictions = inf.predict_strokes(strokes, model, label_decoder)

            for symbol, confidence in predictions:
                if symbol == leader:
                    continue

                if confidence < min_confidence:
                    continue

                current_collisions[symbol].append((key, confidence))
        # Check if there was sufficient overlap for each key found in the collisions.
        for symbol, keys in current_collisions.items():
            if len(keys) / len(samples) > min_overlap:
                # Add only a single entry to the collisions dictionary,
                # using the median confidence.
                confidences = [conf for _, conf in keys]
                median_conf = sorted(confidences)[len(confidences) // 2]
                collisions[symbol].append((leader, median_conf))

    conn.close()

    # Present the results.
    for symbol, collision_list in collisions.items():
        print(f"Symbol: {symbol}")
        for leader, confidence in collision_list:
            print(f"Leader: {leader}, Confidence: {confidence:.2f}")
        print()


if __name__ == "__main__":
    main()
