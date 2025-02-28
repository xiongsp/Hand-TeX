import sqlite3
from pathlib import Path


from training.data_loader import get_data_split, DataSplit, StrokeDataset

split_percentages = {
    DataSplit.TRAIN: 50,
    DataSplit.VALIDATION: 20,
    DataSplit.TEST: 30,
}

split_percentages_no_test = {
    DataSplit.TRAIN: 80,
    DataSplit.VALIDATION: 20,
    DataSplit.TEST: 0,
}


def test_exact_proportions():
    # With 100 samples the computed counts should match the percentages exactly.
    data = list(range(100))
    train = get_data_split(data, DataSplit.TRAIN, split_percentages)
    validation = get_data_split(data, DataSplit.VALIDATION, split_percentages)
    test_ = get_data_split(data, DataSplit.TEST, split_percentages)

    assert len(train) == 50
    assert len(validation) == 20
    assert len(test_) == 30

    # Since the segments are contiguous, verify the slices.
    assert train == data[0:50]
    assert validation == data[50:70]
    assert test_ == data[70:100]

    # Check that the union of splits gives back the original list.
    combined = train + validation + test_
    assert combined == data


def test_minimum_one_sample_each():
    # With exactly 3 samples each split must get exactly one sample.
    data = list(range(3))
    train = get_data_split(data, DataSplit.TRAIN, split_percentages)
    validation = get_data_split(data, DataSplit.VALIDATION, split_percentages)
    test_ = get_data_split(data, DataSplit.TEST, split_percentages)

    assert len(train) >= 1
    assert len(validation) >= 1
    assert len(test_) >= 1

    # With 3 samples and forced one-per-split, the slices are as follows.
    assert train == data[0:1]
    assert validation == data[1:2]
    assert test_ == data[2:3]


def test_no_test_split():
    # With 100 samples and no test split, the computed counts should match the percentages exactly.
    data = list(range(100))
    train = get_data_split(data, DataSplit.TRAIN, split_percentages_no_test)
    validation = get_data_split(data, DataSplit.VALIDATION, split_percentages_no_test)
    test_ = get_data_split(data, DataSplit.TEST, split_percentages_no_test)

    assert len(train) == 80
    assert len(validation) == 20
    assert len(test_) == 0

    # Since the segments are contiguous, verify the slices.
    assert train == data[0:80]
    assert validation == data[80:100]
    assert test_ == []

    # Check that the union of splits gives back the original list.
    combined = train + validation + test_
    assert combined == data


def test_proportional_rounding():
    # With 7 samples the raw ideal counts are:
    #   TRAIN: 7*50/100 = 3.5  -> floor 3, fraction 0.5
    #   VALIDATION: 7*20/100 = 1.4  -> floor 1, fraction 0.4 (but at least 1)
    #   TEST: 7*30/100 = 2.1 -> floor 2, fraction 0.1
    # Sum of floors = 3 + 1 + 2 = 6, so 1 leftover is distributed to TRAIN (highest fraction).
    # Expected counts: TRAIN=4, VALIDATION=1, TEST=2.
    data = list(range(7))
    train = get_data_split(data, DataSplit.TRAIN, split_percentages)
    validation = get_data_split(data, DataSplit.VALIDATION, split_percentages)
    test_ = get_data_split(data, DataSplit.TEST, split_percentages)

    assert len(train) == 4
    assert len(validation) == 1
    assert len(test_) == 2

    # Verify that the segments are contiguous.
    assert train == data[0:4]
    assert validation == data[4:5]
    assert test_ == data[5:7]

    # Verify that the union equals the original data.
    combined = train + validation + test_
    assert combined == data


def create_test_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE samples (id INTEGER PRIMARY KEY, key TEXT, strokes TEXT)")
    # Insert samples for key "A"
    cur.executemany("INSERT INTO samples VALUES (?, ?, ?)", [(i, "A", "") for i in range(1, 21)])
    # Insert at least 3 samples for key "latex2e-|"
    cur.executemany(
        "INSERT INTO samples VALUES (?, ?, ?)", [(i + 100, "latex2e-|", "") for i in range(1, 4)]
    )
    conn.commit()
    conn.close()


# Dummy implementations for the required objects:
class DummySymbolData:
    leaders = ["A"]

    def get_similarity_group(self, key):
        return (key,)

    def all_symbols_to_symbol(self, key):
        return (key,)

    def all_paths_to_symbol(self, key):
        # Yield a single simple path: (symbol_key, transformations, composition)
        yield "A", (), None


class DummyLabelEncoder:
    def transform(self, labels):
        return [0 for _ in labels]


def test_bootstrapping_determinism(tmp_path):
    # Create a temporary SQLite database.
    db_file = tmp_path / "test.db"
    create_test_db(db_file)

    dummy_symbol_data = DummySymbolData()
    dummy_label_encoder = DummyLabelEncoder()

    split_percentages = {
        DataSplit.TRAIN: 80,
        DataSplit.VALIDATION: 1,
        DataSplit.TEST: 19,
    }

    # Build two dataloaders with identical random_seed and split.
    ds1 = StrokeDataset(
        db_path=str(db_file),
        symbol_data=dummy_symbol_data,
        image_size=28,
        label_encoder=dummy_label_encoder,
        random_seed=42,
        split=DataSplit.TRAIN,
        split_percentages=split_percentages,
        random_augmentation=False,
    )
    ds2 = StrokeDataset(
        db_path=str(db_file),
        symbol_data=dummy_symbol_data,
        image_size=28,
        label_encoder=dummy_label_encoder,
        random_seed=42,
        split=DataSplit.TRAIN,
        split_percentages=split_percentages,
        random_augmentation=False,
    )

    # For our purposes, we check that the primary keys (after bootstrapping and splitting) are identical.
    assert ds1.primary_keys == ds2.primary_keys
