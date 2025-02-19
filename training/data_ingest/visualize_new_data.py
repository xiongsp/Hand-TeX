import csv
import json
from collections import defaultdict
from pathlib import Path

import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtSvgWidgets as Qsw
import PySide6.QtWidgets as Qw
import matplotlib.pyplot as plt
from PIL import Image
from loguru import logger

import handtex.data.symbol_metadata
import handtex.symbol_relations as sr
import handtex.utils as ut
import training.image_gen as ig
from handtex.utils import resource_path
from rdp import rdp


def main():
    """
    Load the new data and show each symbol alongside the new drawings.
    """
    symbol_data = sr.SymbolData()
    new_data_dir = Path("../../new_drawings")
    # Load frequency data for the current database.
    frequencies_path = resource_path(handtex.data.symbol_metadata, "symbol_frequency.csv")

    with open(frequencies_path, "r") as file:
        reader = csv.reader(file)
        frequencies = defaultdict(int, {row[0]: int(row[1]) for row in reader})

    total_old = sum(frequencies.values())
    logger.info(f"Loaded {total_old} training set drawings for frequency analysis.")

    # Load new data, gather frequencies from it.
    # All sessions are stored as independant json files.
    new_frequencies = defaultdict(int, {key: 0 for key in symbol_data.all_keys})
    new_drawings: dict[str, list[tuple[str, list[list[tuple[int, int]]]]]] = {}
    for new_file in new_data_dir.glob("*.json"):
        with open(new_file, "r") as file:
            data = json.load(file)
            for drawing in data:
                new_key = drawing["key"]
                new_key = new_key.replace("-OT1", "")
                new_key = new_key.replace("-T1", "")
                if new_key not in new_drawings:
                    new_drawings[new_key] = []
                strokes = drawing["strokes"]
                # strokes = [rdp(stroke, epsilon=4) for stroke in strokes]
                new_drawings[new_key].append((new_file.name, strokes))
                new_frequencies[new_key] += 1

    logger.info(f"Training {len(symbol_data.all_keys)} symbols.")

    total_new = sum(new_frequencies.values())
    logger.info(f"Loaded {total_new} previously recorded drawings for frequency analysis.")

    # Ensure both dicts have the same keys.
    for key in frequencies.keys():
        if key not in new_frequencies:
            new_frequencies[key] = 0
    for key in new_frequencies.keys():
        if key not in frequencies:
            frequencies[key] = 0

    # Visualize the new frequency by stacking it on top of a histogram of the old frequency.
    # Don't add labels.
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    x_labels = list(frequencies.keys())
    x_indices = range(len(x_labels))

    # Plot old frequencies
    plt.bar(
        x_indices,
        frequencies.values(),
        width=bar_width,
        color="blue",
        alpha=0.5,
        label="Old Frequency",
    )
    # Plot new frequencies with a slight offset to align properly
    plt.bar(
        [x + bar_width for x in x_indices],
        new_frequencies.values(),
        width=bar_width,
        color="red",
        alpha=0.7,
        label="New Frequency",
    )

    plt.xlabel("Symbols")
    plt.ylabel("Frequency")
    plt.title("Comparison of Old and New Symbol Frequencies")
    plt.xticks([x + bar_width / 2 for x in x_indices], x_labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    total_drawings = sum(new_frequencies.values())
    newly_trained_symbols = [key for key, value in new_frequencies.items() if value > 0]
    logger.info(
        f"Training {len(newly_trained_symbols)} leader symbols with {total_drawings} new drawings."
    )

    # Visualize the new drawings for each symbol.
    for index, (symbol_key, drawings) in enumerate(new_drawings.items(), start=1):
        app = Qw.QApplication.instance() or Qw.QApplication([])
        mainwindow = Qw.QWidget()
        # We have a drawing of the actual symbol in the top left.
        # Then we have all the new drawings in a grid below that.
        # The grid should be scrollable.
        mainwindow.setWindowTitle(f"Symbol: {symbol_key}")
        mainwindow.setFixedSize(800, 800)
        mainlayout = Qw.QVBoxLayout()
        mainwindow.setLayout(mainlayout)

        # Load the symbol svg.
        # Display it next to the number of new drawings, and the symbol key.
        if symbol_key not in symbol_data:
            # Use the varnothing symbol as a placeholder.
            symbol_key = "amssymb-_varnothing"
        symbol = symbol_data[symbol_key]
        svg_widget = Qsw.QSvgWidget()
        color = app.palette().color(Qg.QPalette.Text).name()
        svg_data = ut.load_symbol_svg(symbol, color)
        svg_widget.load(svg_data)
        svg_widget.renderer().setAspectRatioMode(Qc.Qt.KeepAspectRatio)
        svg_widget.setFixedSize(200, 200)

        symbol_info = Qw.QLabel()
        symbol_info.setText(
            f"Symbol: {symbol_key} ({index}/{len(newly_trained_symbols)})\nNew Drawings: {len(drawings)}"
        )
        font = symbol_info.font()
        font.setPointSize(16)
        symbol_info.setFont(font)
        symbol_info.setWordWrap(True)

        symbol_layout = Qw.QHBoxLayout()
        symbol_layout.addWidget(svg_widget)
        symbol_layout.addWidget(symbol_info)
        symbol_layout.addSpacerItem(Qw.QSpacerItem(0, 0, Qw.QSizePolicy.Policy.Expanding))
        mainlayout.addLayout(symbol_layout)

        # Load the new drawings.
        scroll_area = Qw.QScrollArea()
        scroll_area.setWidgetResizable(True)
        # scroll_area.setFixedHeight(600)
        # scroll_area.setFixedWidth(800)
        scroll_area.setFrameShape(Qw.QFrame.Shape.NoFrame)

        grid = Qw.QGridLayout()
        scroll_widget = Qw.QWidget()
        scroll_widget.setLayout(grid)
        scroll_area.setWidget(scroll_widget)
        mainlayout.addWidget(scroll_area)
        mainlayout.addSpacerItem(Qw.QSpacerItem(0, 0, Qw.QSizePolicy.Policy.Expanding))

        print("--------------------")

        for i, (path, drawing) in enumerate(drawings):
            image = ig.strokes_to_grayscale_image_cv2(drawing, 100)
            image = Image.fromarray(image)
            image = Qg.QImage(
                image.tobytes(),
                image.width,
                image.height,
                image.width,
                Qg.QImage.Format.Format_Grayscale8,
            )
            label = ClickableLabel()
            label.setPixmap(Qg.QPixmap.fromImage(image))
            grid.addWidget(label, i // 7, i % 7)
            # When clicked, show the path and drawing code.
            # Ensure the text is selectable.
            label.clicked.connect(lambda: print(path, drawing))
            # Print number of points in the drawing.
            print(f"{sum(len(stroke) for stroke in drawing)} points.")

        mainwindow.show()

        app.exec()


class ClickableLabel(Qw.QLabel):
    clicked = Qc.Signal()  # Define a custom signal

    def mousePressEvent(self, event):
        if event.button() == Qc.Qt.LeftButton:
            self.clicked.emit()  # Emit the signal when the label is clicked
        super().mousePressEvent(event)


if __name__ == "__main__":
    main()
