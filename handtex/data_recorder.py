import os
import json
import csv
from pathlib import Path
import random

from loguru import logger

import handtex.structures as st


class DataRecorder:

    symbols: dict[str, st.Symbol]
    frequencies: dict[str, int]
    data_dir: Path
    current_data: list[st.SymbolDrawing]

    # Manage loading/saving symbols for new training data generation.
    def __init__(self, symbols: dict[str, st.Symbol]):
        self.current_data = []

        # Load the new data location from environment variables.
        if "NEW_DATA_DIR" in os.environ:
            self.data_dir = Path(os.environ["NEW_DATA_DIR"])
        else:
            self.data_dir = Path("new_data").absolute()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"New data location: {self.data_dir}")

        self.symbols = symbols

        # Load old frequencies.
        # Load the symbol frequency file from environment variables.
        if "SYMBOL_FREQUENCY" in os.environ:
            old_frequencies_path = Path(os.environ["SYMBOL_FREQUENCY"])
        else:
            old_frequencies_path = Path("symbol_frequency.csv").absolute()
        if not old_frequencies_path.exists():
            raise FileNotFoundError(f"Could not find {old_frequencies_path}")

        with open(old_frequencies_path, "r") as file:
            reader = csv.reader(file)
            self.frequencies = {row[0]: int(row[1]) for row in reader}

        total_old = sum(self.frequencies.values())
        logger.info(f"Loaded {total_old} training set drawings for frequency analysis.")

        # Load new data, gather frequencies from it.
        # All sessions are stored as independant json files.
        new_frequencies = {key: 0 for key in self.symbols.keys()}
        for new_file in self.data_dir.glob("*.json"):
            with open(new_file, "r") as file:
                data = json.load(file)
                for drawing in data:
                    new_frequencies[drawing["key"]] += 1

        total_new = sum(new_frequencies.values())
        logger.info(f"Loaded {total_new} previously recorded drawings for frequency analysis.")

        # Combine the old and new frequencies.
        for key, value in new_frequencies.items():
            self.frequencies[key] += value
        logger.info(f"Total of {sum(self.frequencies.values())} drawings for frequency analysis.")

    def select_symbol(self, bias: float = 0.5) -> str:
        """
        Select a symbol to draw based on the frequency of the symbol in the training set.
        The bias parameter can be used to skew the selection towards less common symbols.
        0 Bias will select symbols uniformly, 1 Bias will select the least common symbols.

        :param bias: The bias towards less common symbols.
        :return: The symbol's key.
        """
        # Calculate the probability of selecting a symbol.
        symbol_keys = list(self.symbols.keys())
        symbol_weights = []
        for key in symbol_keys:
            weight = (1 / self.frequencies[key]) ** bias
            symbol_weights.append(weight)
        return random.choices(symbol_keys, weights=symbol_weights)[0]

    def get_symbol_rarity(self, key: str) -> int:
        """
        Get the rarity of a symbol based on the frequency of the symbol in the training set.
        The rarity is the rarity index, giving a higher value to less common symbols.

        :param key: The symbol's key.
        :return: The rarity of the symbol.
        """
        max_rarity = max(self.frequencies.values())
        return round(1000 * (1 - self.frequencies[key] / max_rarity))

    def get_symbol_sample_count(self, key: str) -> int:
        """
        Get the number of samples for a symbol.

        :param key: The symbol's key.
        :return: The number of samples for the symbol.
        """
        return self.frequencies[key]

    def submit_drawing(self, drawing: st.SymbolDrawing) -> None:
        """
        Submit a drawing to the data recorder.

        :param drawing: The drawing to submit.
        """
        self.current_data.append(drawing)
        logger.info(f"Recorded drawing for symbol {drawing.key}.")
        logger.info(f"Recorded drawing for symbol {drawing.strokes}.")

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
