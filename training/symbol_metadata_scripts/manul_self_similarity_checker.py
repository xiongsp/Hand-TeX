import csv
import numpy as np
import os

import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtSvg as Qs
import PySide6.QtSvgWidgets as Qsw
import PySide6.QtWidgets as Qw

import handtex.utils as ut
import handtex.symbol_relations as sr


def reflection_matrix(angle: float, image_size: int = 100) -> Qg.QTransform:
    """
    Reflect the stroke data along an axis that passes through the center of the image
    at a specified angle. The center is at coordinates (image_size / 2, image_size / 2).
    Positive angles rotate the axis counter-clockwise.

    :param angle: Angle in degrees defining the axis to reflect across.
    :param image_size: Size of the image the strokes are drawn on.
    :return: QTransform reflection matrix.
    """
    x_offset = image_size / 2
    y_offset = image_size / 2

    transformation = Qg.QTransform()
    # Translate to center
    transformation.translate(x_offset, y_offset)
    # Rotate to align reflection axis with x-axis
    transformation.rotate(angle)
    # Reflect across x-axis (scale y by -1)
    transformation.scale(1, -1)
    # Rotate back to original orientation
    transformation.rotate(-angle)
    # Translate back to original position
    transformation.translate(-x_offset, -y_offset)

    return transformation


def main():
    """
    Show various transformations of a symbol, allowing the user to select valid similarities.
    """
    # Set environment variable to disable the bounding rect check in Qt SVG handler
    os.environ["QT_SVG_DEFAULT_OPTIONS"] = "2"

    symbols = sr.load_symbols()

    leader_frequency_path = "../../handtex/data/symbol_metadata/augmented_symbol_frequency.csv"
    with open(leader_frequency_path, "r") as file:
        leader_frequencies = {row[0]: int(row[1]) for row in csv.reader(file)}

    # Get list list of symbols, sorted ascending by frequency.
    sorted_symbols = sorted(leader_frequencies.keys(), key=leader_frequencies.get)
    print(len(sorted_symbols))

    start_at = "amssymb-OT1-_mathcal{X}"
    if start_at:
        start_at_index = sorted_symbols.index(start_at)
        sorted_symbols = sorted_symbols[start_at_index:]

    # Visualize the new drawings for each symbol.
    app = Qw.QApplication.instance() or Qw.QApplication([])
    for index, symbol_key in enumerate(sorted_symbols, start=1):
        mainwindow = Qw.QWidget()
        # We have a drawing of the actual symbol in the top left.
        # Then we have all the new drawings in a grid below that.
        # The grid should be scrollable.
        mainwindow.setWindowTitle(f"Symbol: {symbol_key}")
        mainwindow.setFixedSize(900, 700)
        mainlayout = Qw.QVBoxLayout()
        mainwindow.setLayout(mainlayout)

        # Load the symbol svg.
        # Display it next to the number of new drawings, and the symbol key.
        symbol = symbols[symbol_key]
        svg_widget = Qsw.QSvgWidget()
        color = app.palette().color(Qg.QPalette.Text).name()
        svg_data = ut.load_symbol_svg(symbol, color)
        svg_widget.load(svg_data)
        svg_widget.renderer().setAspectRatioMode(Qc.Qt.KeepAspectRatio)
        svg_widget.setFixedSize(200, 200)

        symbol_info = Qw.QLabel()
        symbol_info.setText(f"Symbol: {symbol_key} ({index}/{len(sorted_symbols)})")
        font = symbol_info.font()
        font.setPointSize(16)
        symbol_info.setFont(font)

        symbol_layout = Qw.QHBoxLayout()
        symbol_layout.addWidget(svg_widget)
        symbol_layout.addWidget(symbol_info)
        symbol_layout.addSpacerItem(Qw.QSpacerItem(0, 0, Qw.QSizePolicy.Policy.Expanding))
        mainlayout.addLayout(symbol_layout)

        # Load the new drawings.
        scroll_area = Qw.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(Qw.QFrame.Shape.NoFrame)

        grid = Qw.QGridLayout()
        scroll_widget = Qw.QWidget()
        scroll_widget.setLayout(grid)
        scroll_area.setWidget(scroll_widget)
        mainlayout.addWidget(scroll_area)
        mainlayout.addSpacerItem(Qw.QSpacerItem(0, 0, Qw.QSizePolicy.Policy.Expanding))

        button_list = []

        # Draw 7 copies of the symbol, each with a rotation of 45 degrees.
        for i in range(1, 8):
            # Insert the SVG image, but rotated.
            angle = i * 45
            # Render to a 100x100 pixmap.
            pixmap = Qg.QPixmap(100, 100)
            pixmap.fill(Qc.Qt.GlobalColor.transparent)
            base_renderer = Qs.QSvgRenderer(svg_data)
            base_renderer.setAspectRatioMode(Qc.Qt.AspectRatioMode.KeepAspectRatio)
            painter = Qg.QPainter(pixmap)
            painter.translate(50, 50)
            painter.rotate(-angle)
            painter.translate(-50, -50)
            base_renderer.render(painter)
            painter.end()

            # Create a checkable label.
            button = Qw.QPushButton()
            button.setCheckable(True)
            button.setIcon(Qg.QIcon(pixmap))
            button.setIconSize(Qc.QSize(100, 100))
            button.setToolTip(f"Rotation: {angle}")
            button.setStyleSheet(
                "QPushButton:checked { background-color: %s }"
                % app.palette().color(Qg.QPalette.Highlight).name()
            )
            button.data = f"rot{angle}"
            button_list.append(button)
            grid.addWidget(button, 1, i)

        # Draw 8 copies of the symbol, each flipped around the central axis at increments of 22.5 degrees.
        for i in range(4):
            # Insert the SVG image, but flipped using a reflection matrix.
            angle = i * 45
            transformation = reflection_matrix(180 - angle, 100)
            # Render to a 100x100 pixmap.
            pixmap = Qg.QPixmap(100, 100)
            pixmap.fill(Qc.Qt.GlobalColor.transparent)
            base_renderer = Qs.QSvgRenderer(
                svg_data
            )  # Create a new renderer for each transformation
            base_renderer.setAspectRatioMode(Qc.Qt.AspectRatioMode.KeepAspectRatio)
            painter = Qg.QPainter(pixmap)
            painter.setTransform(transformation, combine=True)
            base_renderer.render(painter)
            painter.end()

            # Create a checkable label.
            button = Qw.QPushButton()
            button.setCheckable(True)
            button.setIcon(Qg.QIcon(pixmap))
            button.setIconSize(Qc.QSize(100, 100))
            button.setToolTip(f"Flip Angle: {angle}")
            button.setStyleSheet(
                "QPushButton:checked { background-color: %s }"
                % app.palette().color(Qg.QPalette.Highlight).name()
            )
            button.data = f"mir{angle}"
            button_list.append(button)
            grid.addWidget(button, 2, i + 1)

        mainwindow.show()
        app.exec()
        # Gather all button data from checked buttons.
        # Format: symbol: transformation1 transformation2 ...
        selected_transformations = []
        for button in button_list:
            if button.isChecked():
                selected_transformations.append(button.data)
        if not selected_transformations:
            continue
        symmetry_data = f"{symbol_key}: " + " ".join(selected_transformations)
        print(symmetry_data)


if __name__ == "__main__":
    main()
