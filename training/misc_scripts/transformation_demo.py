import json
import sqlite3

import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtSvgWidgets as Qsw
import PySide6.QtWidgets as Qw
import numpy as np
from PIL import Image

import handtex.utils as ut
import handtex.symbol_relations as sr
import training.image_gen as ig


def main():
    """
    Load the new data and show each symbol alongside the new drawings.
    """
    symbols = sr.load_symbols()
    test_set = (
        "latex2e-_rightarrow",
        "latex2e-_int",
        "latex2e-_mid",
        "latex2e-_sum",
        "latex2e-_alpha",
        "latex2e-_cdot",
        "amssymb-_lll",
    )
    db_path = "../database/handtex.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # # time the new vs the old apply_transformations function.
    # # Use a random dataset, and apply the transformations to it.
    # # Then visualize the results.
    # import time
    # import random
    #
    # test_strokes = []
    # for key in symbols.keys():
    #     cursor.execute("SELECT strokes FROM samples WHERE key = ?", (key,))
    #     strokes = json.loads(cursor.fetchone()[0])
    #     test_strokes.append(strokes)

    # key = "latex2e-_lambda"
    # cursor.execute("SELECT strokes FROM samples WHERE key = ?", (key,))
    # test_strokes = [json.loads(s[0]) for s in cursor.fetchall()]

    # transformation = [ig.rotation_matrix(-45), ig.skew_matrix(0.5, 0)]
    # # Test the old apply_transformations function.
    # start = time.time()
    # for strokes in test_strokes:
    #     ig.strokes_to_grayscale_image_cv2(strokes, image_size=48)
    # old_time = time.time() - start

    # # Test the new apply_transformations function.
    # start = time.time()
    # for strokes in test_strokes:
    #     ig.apply_transformations(strokes, transformation)
    # new_time = time.time() - start

    # print(f"Old time: {(old_time/len(test_strokes))*1000:.4f} ms")
    # print(f"New time: {(new_time/len(test_strokes))*1000:.4f} ms")

    # Visualize the new drawings for each symbol.
    for index, symbol_key in enumerate(test_set, start=1):
        app = Qw.QApplication.instance() or Qw.QApplication([])
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
        symbol_info.setText(f"Symbol: {symbol_key} ({index}/{len(test_set)})")
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
        # scroll_area.setFixedHeight(600)
        # scroll_area.setFixedWidth(800)
        scroll_area.setFrameShape(Qw.QFrame.Shape.NoFrame)

        grid = Qw.QGridLayout()
        scroll_widget = Qw.QWidget()
        scroll_widget.setLayout(grid)
        scroll_area.setWidget(scroll_widget)
        mainlayout.addWidget(scroll_area)
        mainlayout.addSpacerItem(Qw.QSpacerItem(0, 0, Qw.QSizePolicy.Policy.Expanding))

        # Transform the strokes.
        cursor.execute("SELECT strokes FROM samples WHERE key = ?", (symbol_key,))
        strokes = json.loads(cursor.fetchone()[0])

        # Create 8 drawings for each symbol rotation.
        drawings_rot = []
        for i in range(0, 360, 45):
            # drawings_rot.append(ig.rotate_strokes(strokes, -i, 1000))
            drawings_rot.append(
                ig.apply_transformations(
                    strokes,
                    [ig.rotation_matrix(i)],
                    shuffle_transformations=True,
                )
            )

        for i, drawing in enumerate(drawings_rot):
            image = ig.strokes_to_grayscale_image_cv2(drawing, 100)
            image = Image.fromarray(image)
            image = Qg.QImage(
                image.tobytes(),
                image.width,
                image.height,
                image.width,
                Qg.QImage.Format.Format_Grayscale8,
            )
            label = Qw.QLabel()
            label.setPixmap(Qg.QPixmap.fromImage(image))
            label.setToolTip(f"Rotation: {i*45}")
            grid.addWidget(label, 0, i % 8)

        # Now 8 for mirroring the symbol.
        drawings_mir = []
        for i in np.linspace(0, 180 - 180 / 8, 8):
            # drawings_mir.append(ig.mirror_strokes(strokes, -i, 1000))
            drawings_mir.append((i, ig.apply_transformations(strokes, ig.reflection_matrix(i))))

        for i, (angle, drawing) in enumerate(drawings_mir):
            image = ig.strokes_to_grayscale_image_cv2(drawing, 100)
            image = Image.fromarray(image)
            image = Qg.QImage(
                image.tobytes(),
                image.width,
                image.height,
                image.width,
                Qg.QImage.Format.Format_Grayscale8,
            )
            label = Qw.QLabel()
            label.setPixmap(Qg.QPixmap.fromImage(image))
            label.setToolTip(f"Mirror: {angle}")
            grid.addWidget(label, 1, i % 8)

        # Now 8 for scaling the symbol.
        # 4 for each direction.
        drawings_sca = []
        for i in np.linspace(0.9, 0.5, 4):
            # drawings_sca.append((i, ig.scale_strokes(strokes, i, 1, 1000)))
            drawings_sca.append((i, ig.apply_transformations(strokes, ig.scale_matrix(i, 1, 1000))))
        for i in np.linspace(0.9, 0.5, 4):
            # drawings_sca.append((i, ig.scale_strokes(strokes, 1, i, 1000)))
            drawings_sca.append((i, ig.apply_transformations(strokes, ig.scale_matrix(i, 1, 1000))))

        for i, (scale, drawing) in enumerate(drawings_sca):
            image = ig.strokes_to_grayscale_image_cv2(drawing, 100)
            image = Image.fromarray(image)
            image = Qg.QImage(
                image.tobytes(),
                image.width,
                image.height,
                image.width,
                Qg.QImage.Format.Format_Grayscale8,
            )
            label = Qw.QLabel()
            label.setPixmap(Qg.QPixmap.fromImage(image))
            label.setToolTip(f"Scale: {scale}")
            grid.addWidget(label, 2, i % 8)

        # Now 8 for skewing the symbol.
        # 4 for each direction.
        drawings_ske = []
        for i in np.linspace(0.2, 0.9, 4):
            # drawings_ske.append((i, ig.skew_strokes(strokes, i, 0, 1000)))
            drawings_ske.append((i, ig.apply_transformations(strokes, ig.skew_matrix(i, 0, 1000))))
        for i in np.linspace(0.2, 0.9, 4):
            # drawings_ske.append((i, ig.skew_strokes(strokes, 0, i, 1000)))
            drawings_ske.append((i, ig.apply_transformations(strokes, ig.skew_matrix(0, i, 1000))))

        for i, (skew, drawing) in enumerate(drawings_ske):
            image = ig.strokes_to_grayscale_image_cv2(drawing, 100)
            image = Image.fromarray(image)
            image = Qg.QImage(
                image.tobytes(),
                image.width,
                image.height,
                image.width,
                Qg.QImage.Format.Format_Grayscale8,
            )
            label = Qw.QLabel()
            label.setPixmap(Qg.QPixmap.fromImage(image))
            label.setToolTip(f"Skew: {skew}")
            grid.addWidget(label, 3, i % 8)

        mainwindow.show()

        app.exec()

    conn.close()


if __name__ == "__main__":
    main()
