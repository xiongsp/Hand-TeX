import os
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel, QScrollArea, QGridLayout
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QPalette, QColor
from pathlib import Path

# Set environment variable to avoid Wayland issues
os.environ["QT_QPA_PLATFORM"] = "xcb"

# List of SVG files to be rendered
# Grab all files inside the "symbols" directory
svg_files = [file.name for file in Path("symbols").iterdir()]

# PySide6 Application
app = QApplication([])
window = QWidget()
layout = QVBoxLayout()

# Set the background color to purple
palette = QPalette()
palette.setColor(QPalette.Window, QColor("#9966FF"))
window.setPalette(palette)

# Scroll Area to hold the SVG widgets
scroll_area = QScrollArea()
scroll_area_widget = QWidget()
scroll_area_layout = QGridLayout()  # Use QGridLayout for arranging widgets in a grid

# Generate QSvgWidgets for each SVG file and add them to the grid layout
row = 0
col = 0
max_columns = 5  # Define how many icons you want per row

for svg_file in svg_files:
    svg_file_path = "../../handtex/data/symbols/" + svg_file
    if os.path.exists(svg_file_path):
        # Load the SVG content and modify the fill color
        with open(svg_file_path, "r") as file:
            svg_content = file.read()

        modified_svg_content = svg_content.replace('stroke="#000000"', 'stroke="#225555"')
        modified_svg_content = modified_svg_content.replace('stroke="#000"', 'stroke="#225555"')
        modified_svg_content = modified_svg_content.replace(
            'fill="#000000"', 'fill="#225555"'
        )  # Replace fill color with red (#FF0000)

        # Convert modified SVG content to QByteArray
        svg_byte_array = QByteArray(modified_svg_content.encode("utf-8"))

        # Create QSvgWidget from QByteArray
        svg_widget = QSvgWidget()
        svg_widget.load(svg_byte_array)
        svg_widget.setFixedSize(100, 100)  # Increase size for better visibility
        svg_widget.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
        svg_widget.setToolTip(svg_file)  # Set tooltip to reveal the SVG file name
        scroll_area_layout.addWidget(svg_widget, row, col)

        # Update column and row to arrange in grid
        col += 1
        if col >= max_columns:
            col = 0
            row += 1
    else:
        # In case the SVG file does not exist, add a label with a warning message.
        label = QLabel(f"File not found: {svg_file_path}")
        label.setAlignment(Qt.AlignCenter)
        scroll_area_layout.addWidget(label, row, col)
        col += 1
        if col >= max_columns:
            col = 0
            row += 1

scroll_area_widget.setLayout(scroll_area_layout)
scroll_area.setWidget(scroll_area_widget)
scroll_area.setWidgetResizable(True)

# Add scroll area to main layout
layout.addWidget(scroll_area)
window.setLayout(layout)

window.setWindowTitle("SVG Symbols in PySide6")
window.resize(800, 600)
window.show()
app.exec()
