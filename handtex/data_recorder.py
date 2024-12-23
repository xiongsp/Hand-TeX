import csv
import datetime
import json
import random
from collections import defaultdict
from importlib import resources
from pathlib import Path

from PySide6.QtCore import Signal
from loguru import logger

import handtex.config as cfg
import handtex.data.symbol_metadata
import handtex.structures as st
import handtex.symbol_relations as sr
import handtex.utils as ut


class DataRecorder:

    symbol_data: sr.SymbolData
    config: cfg.Config
    save_path: Path
    current_data: list[st.SymbolDrawing]
    frequencies: defaultdict[str, int]
    augmented_frequencies: defaultdict[str, int]

    last_100_symbols: list[str]

    has_submissions: Signal

    # Manage loading/saving symbols for new training data generation.
    def __init__(
        self,
        symbol_data: sr.SymbolData,
        has_submissions: Signal,
        new_data_dir: str = "",
    ):
        self.current_data = []
        self.symbol_data = symbol_data
        self.frequencies = defaultdict(int)
        self.augmented_frequencies = defaultdict(int)
        self.has_submissions = has_submissions

        # Don't randomly select one of the last 20 symbols.
        self.last_100_symbols = []

        data_dir = Path(new_data_dir).absolute()
        self.save_path = Path("unset")
        self.set_new_data_dir(data_dir)

    def set_new_data_dir(self, data_dir: Path) -> None:
        """
        Load the present frequencies and set the new data directory.

        :param data_dir: The new data directory.
        """
        # Load the new data location from environment variables.
        logger.info(f"New data location: {data_dir}")

        # Load frequency data for the current database.
        with resources.path(handtex.data.symbol_metadata, "symbol_frequency.csv") as path:
            frequencies_path = path

        with resources.path(handtex.data.symbol_metadata, "augmented_symbol_frequency.csv") as path:
            augmented_frequencies_path = path

        with open(frequencies_path, "r") as file:
            reader = csv.reader(file)
            frequencies = defaultdict(int, {row[0]: int(row[1]) for row in reader})

        with open(augmented_frequencies_path, "r") as file:
            reader = csv.reader(file)
            augmented_frequencies = defaultdict(int, {row[0]: int(row[1]) for row in reader})

        total_old = sum(frequencies.values())
        logger.info(f"Loaded {total_old} training set drawings for frequency analysis.")

        # Load new data, gather frequencies from it.
        # All sessions are stored as independant json files.
        new_frequencies = {key: 0 for key in self.symbol_data.all_keys}
        for new_file in data_dir.glob("*.json"):
            with open(new_file, "r") as file:
                data = json.load(file)
                for drawing in data:
                    new_key = drawing["key"]
                    # Add 1 to all symbols in it's symmetry group.
                    # This is the collection of ancestors for it's leader.
                    if new_key not in self.symbol_data.leaders:
                        new_key = self.symbol_data.to_leader[new_key]
                    for key in self.symbol_data.all_symbols_to_symbol(new_key):
                        new_frequencies[key] += 1

        logger.info(f"Training {len(self.symbol_data.all_keys)} symbols.")

        total_new = sum(new_frequencies.values())
        logger.info(f"Loaded {total_new} previously recorded drawings for frequency analysis.")

        # Combine the old and new frequencies.
        for key, value in new_frequencies.items():
            frequencies[key] += value

        self.frequencies = frequencies
        self.augmented_frequencies = augmented_frequencies

        logger.info(f"Total of {sum(self.frequencies.values())} drawings for frequency analysis.")

        # Assign the save path.
        self.save_path = (
            data_dir / f"session_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        self.save_path = ut.ensure_unique_file_path(self.save_path)
        logger.info(f"Session data will be saved to {self.save_path}.")

    def select_symbol(self, bias: float = 0.5) -> str:
        """
        Select a symbol to draw based on the frequency of the symbol in the training set.
        The bias parameter can be used to skew the selection towards less common symbols.
        0 Bias will select symbols uniformly, 1 Bias will select the least common symbols.

        :param bias: The bias towards less common symbols.
        :return: The symbol's key.
        """
        # Calculate the probability of selecting a symbol.
        symbol_keys = self.symbol_data.all_keys
        symbol_weights = []
        for key in symbol_keys:
            balanced_frequency = (self.frequencies[key] + self.augmented_frequencies[key] + 1) / 2
            weight = (1 / balanced_frequency) ** bias
            symbol_weights.append(weight)
        return random.choices(symbol_keys, weights=symbol_weights)[0]

    def get_symbol_rarity(self, key: str) -> int | None:
        """
        Get the rarity of a symbol based on the frequency of the symbol in the training set.
        The rarity is the percentile of symbols that outrank it.

        :param key: The symbol's key.
        :return: The rarity of the symbol.
        """
        averaged_frequencies = {
            key: (1 + self.frequencies[key] + self.augmented_frequencies[key]) / 2
            for key in self.frequencies
        }
        sorted_symbols = sorted(averaged_frequencies.items(), key=lambda item: item[1])
        total_symbols = len(sorted_symbols)

        target_rank = None
        for rank, (symbol_key, frequency) in enumerate(sorted_symbols):
            if symbol_key == key:
                target_rank = rank
                break

        if target_rank is None:
            return 9001

        percentile = round((1 - (target_rank / total_symbols)) * 100)

        return percentile

    def get_symbol_sample_count(self, key: str) -> int:
        """
        Get the number of samples for a symbol.

        :param key: The symbol's key.
        :return: The number of samples for the symbol.
        """
        return self.frequencies[key]

    def undo_submission(self) -> st.SymbolDrawing | None:
        """
        Undo the last submission, returning the drawing.

        :return: The last submitted drawing.
        """
        if self.current_data:
            drawing = self.current_data.pop()
            self.frequencies[drawing.key] -= 1
            logger.info(f"Undid submission for symbol {drawing.key}.")
            self.has_submissions.emit(bool(self.current_data))
            return drawing

        self.has_submissions.emit(False)

        self.save_data()

    def submit_drawing(self, drawing: st.SymbolDrawing) -> None:
        """
        Submit a drawing to the data recorder.

        :param drawing: The drawing to submit.
        """
        logger.info(
            f"Recorded drawing for symbol {drawing.key} "
            f"(scale: {drawing.scaling}, offset: {drawing.x_offset}, {drawing.y_offset})."
        )
        # logger.info(f"Recorded drawing for symbol {drawing.strokes}.")

        self.current_data.append(drawing)
        self.frequencies[drawing.key] += 1

        self.has_submissions.emit(bool(self.current_data))

        self.save_data()

        # Debug:
        # def plot_strokes(strokes):
        #     from matplotlib import pyplot as plt
        #
        #     fig, axes = plt.subplots(2, 1, figsize=(5, 10))
        #     # Plot original strokes with points
        #     axes[0].set_title("Original Strokes (With Points)")
        #     for stroke in strokes:
        #         stroke_x = [point[0] for point in stroke]
        #         stroke_y = [point[1] for point in stroke]
        #         axes[0].plot(stroke_x, stroke_y, marker="o")
        #
        #     # Plot original strokes without points
        #     for stroke in strokes:
        #         stroke_x = [point[0] for point in stroke]
        #         stroke_y = [point[1] for point in stroke]
        #         axes[1].plot(stroke_x, stroke_y)
        #
        #     for i in range(2):
        #         axes[i].axis("equal")
        #         # Don't invert the y-axis
        #         axes[i].set_xlabel("X Coordinate")
        #         axes[i].set_ylabel("Y Coordinate")
        #         axes[i].set_xlim(0, 1000)
        #         axes[i].set_ylim(1000, 0)
        #         axes[i].set_aspect("equal", adjustable="box")
        #
        #     plt.tight_layout()
        #     plt.show()
        #
        # plot_strokes(drawing.strokes)

    def save_data(self) -> None:
        """
        Save the collected data to a json file.
        Overwrite the same file if it already exists.
        """
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as file:
            # Use compact json.
            json.dump(
                [drawing.dump() for drawing in self.current_data], file, separators=(",", ":\n")
            )

        logger.info(f"Saved session data to {self.save_path}.")
