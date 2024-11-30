import os

import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtSvg as Qs
import PySide6.QtSvgWidgets as Qsw
import PySide6.QtWidgets as Qw

import handtex.symbol_relations as sr
import handtex.utils as ut


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


def rotation_matrix(angle: float, image_size: int = 100) -> Qg.QTransform:
    """
    Rotate the stroke data around the center of the image by a specified angle.
    The center is at coordinates (image_size / 2, image_size / 2).
    Positive angles rotate counter-clockwise.

    :param angle: Angle in degrees to rotate the stroke data by.
    :param image_size: Size of the image the strokes are drawn on.
    :return: QTransform rotation matrix.
    """
    x_offset = image_size / 2
    y_offset = image_size / 2

    transformation = Qg.QTransform()
    # Translate to center
    transformation.translate(x_offset, y_offset)
    # Rotate around center
    transformation.rotate(angle)
    # Translate back to original position
    transformation.translate(-x_offset, -y_offset)

    return transformation


def main():
    """
    Show existing symmetry info and help creating better associations.
    """
    # raise QtTransformationsAreAnnoyingError("the mirroring isn't properly applied after rotation")
    # Set environment variable to disable the bounding rect check in Qt SVG handler
    os.environ["QT_SVG_DEFAULT_OPTIONS"] = "2"

    symbol_data = sr.SymbolData()

    app = Qw.QApplication.instance() or Qw.QApplication([])

    symbols_to_inspect = []
    for symbol in symbol_data.leaders:
        # If the symbol has any paths with a transformation, add it to the list.
        if any(
            transformations for path, transformations in symbol_data.all_paths_to_symbol(symbol)
        ):
            symbols_to_inspect.append(symbol)
    # Manual override:
    symbols_to_inspect = [key for key in symbol_data.leaders if "arrow" in key]

    for index, symbol in enumerate(symbols_to_inspect, start=1):
        mainwindow = Qw.QWidget()
        # We have a drawing of the actual symbol in the top left.
        # Then we have all the new drawings in a grid below that.
        # The grid should be scrollable.
        mainwindow.setWindowTitle(f"Symbol: {symbol} ({index}/{len(symbols_to_inspect)})")
        mainlayout = Qw.QVBoxLayout()
        mainwindow.setLayout(mainlayout)

        paths = symbol_data.all_paths_to_symbol(symbol)

        # Load the symbol svg.
        # Display it next to the number of new drawings, and the symbol key.
        header_layout = Qw.QHBoxLayout()
        symbol = symbol_data.symbol_data[symbol]
        svg_widget = Qsw.QSvgWidget()
        color = app.palette().color(Qg.QPalette.Text).name()
        svg_data = ut.load_symbol_svg(symbol, color)
        svg_widget.load(svg_data)
        svg_widget.renderer().setAspectRatioMode(Qc.Qt.KeepAspectRatio)
        svg_widget.setFixedSize(200, 200)

        symbol_info = Qw.QLabel()
        symbol_info.setText(f"{symbol} ({len(paths)} paths)")
        symbol_info.setTextInteractionFlags(Qc.Qt.TextInteractionFlag.TextSelectableByMouse)
        font = symbol_info.font()
        font.setPointSize(16)
        symbol_info.setFont(font)

        header_layout.addWidget(svg_widget)
        header_layout.addWidget(symbol_info)
        mainlayout.addLayout(header_layout)

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

        last_symbol = paths[0][0]
        current_row = 0
        current_column = 0
        # For each path, draw the resulting symbol.
        for symbol_from, transformations in paths:
            symbol_from_data = symbol_data.symbol_data[symbol_from]
            symbol_from_svg = ut.load_symbol_svg(symbol_from_data, color)
            # Render to a 100x100 pixmap.
            pixmap = Qg.QPixmap(100, 100)
            pixmap.fill(Qc.Qt.GlobalColor.transparent)
            base_renderer = Qs.QSvgRenderer(symbol_from_svg)
            base_renderer.setAspectRatioMode(Qc.Qt.AspectRatioMode.KeepAspectRatio)
            painter = Qg.QPainter(pixmap)
            matrices = []
            for transformation in transformations:
                if transformation.is_rotation:
                    angle = transformation.angle
                    matrices.append(rotation_matrix(-angle, 100))
                else:
                    angle = transformation.angle
                    matrices.append(reflection_matrix(180 - angle, 100))
            final_matrix = Qg.QTransform()
            for matrix in matrices:
                final_matrix *= matrix
            painter.setTransform(final_matrix)
            base_renderer.render(painter)
            painter.end()

            widget = Qw.QLabel()
            widget.setPixmap(pixmap)
            trans_str = " ".join(map(str, transformations))
            widget.setToolTip(f"Transform {symbol_from}: {trans_str}")
            # Arrange the widgets in a grid of 6 columns.
            # But go to a new line if the symbol changes.
            if current_column == 12 or last_symbol != symbol_from:
                current_row += 1
                current_column = 0
                last_symbol = symbol_from
            current_column += 1
            grid.addWidget(widget, current_row, current_column)

        mainwindow.showMaximized()
        app.exec()
        # Gather all button data from checked buttons.
        # Format: symbol: transformation1 transformation2 ...


if __name__ == "__main__":
    main()
