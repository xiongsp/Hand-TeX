import csv
import numpy as np
import os
import re

import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtSvg as Qs
import PySide6.QtSvgWidgets as Qsw
import PySide6.QtWidgets as Qw

import handtex.utils as ut
import handtex.symbol_relations as sr
import handtex.structures as st


def main():
    """
    Show existing negation strategy and allow user to select a better one.
    """
    # Set environment variable to disable the bounding rect check in Qt SVG handler
    os.environ["QT_SVG_DEFAULT_OPTIONS"] = "2"

    symbols = sr.load_symbols()

    symmetry_path = "../../handtex/data/symbol_metadata/negations.txt"
    symmetries: list[tuple[str, st.Negation, str]] = []
    # Example: latex2e-OT1-_lfloor -/ rot22 o0 /- latex2e-OT1-_lnot
    pattern = re.compile(r"(\S+) -/ *(.*?) */- (\S+)")
    with open(symmetry_path, "r") as file:
        for line in file.readlines():
            match = pattern.match(line)
            if match:
                symmetries.append(
                    (match.group(1), st.Negation.from_string(match.group(2)), match.group(3))
                )

    app = Qw.QApplication.instance() or Qw.QApplication([])
    for index, (s_from, negation, s_to) in enumerate(symmetries, start=1):
        mainwindow = Qw.QWidget()
        # We have a drawing of the actual symbol in the top left.
        # Then we have all the new drawings in a grid below that.
        # The grid should be scrollable.
        mainwindow.setWindowTitle(f"Symbol: {s_from}")
        mainwindow.setFixedSize(900, 700)
        mainlayout = Qw.QVBoxLayout()
        mainwindow.setLayout(mainlayout)

        # Load the symbol svg.
        # Display it next to the number of new drawings, and the symbol key.
        symbol = symbols[s_from]
        svg_widget = Qsw.QSvgWidget()
        color = app.palette().color(Qg.QPalette.Text).name()
        svg_data = ut.load_symbol_svg(symbol, color)
        svg_widget.load(svg_data)
        svg_widget.renderer().setAspectRatioMode(Qc.Qt.KeepAspectRatio)
        svg_widget.setFixedSize(200, 200)

        symbol2 = symbols[s_to]
        svg_widget2 = Qsw.QSvgWidget()
        svg_data2 = ut.load_symbol_svg(symbol2, color)
        svg_widget2.load(svg_data2)
        svg_widget2.renderer().setAspectRatioMode(Qc.Qt.KeepAspectRatio)
        svg_widget2.setFixedSize(200, 200)

        symbol_info = Qw.QLabel()
        symbol_info.setText(f"{s_from} --> \n{s_to}\n({index}/{len(symmetries)})")
        font = symbol_info.font()
        font.setPointSize(16)
        symbol_info.setFont(font)

        symbol_layout = Qw.QHBoxLayout()
        symbol_layout.addWidget(svg_widget)
        symbol_layout.addWidget(symbol_info)
        symbol_layout.addWidget(svg_widget2)
        symbol_layout.addSpacerItem(Qw.QSpacerItem(0, 0, Qw.QSizePolicy.Policy.Expanding))
        mainlayout.addLayout(symbol_layout)

        # Load the new drawings.
        scroll_area = Qw.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(Qw.QFrame.Shape.NoFrame)

        # grid = Qw.QGridLayout()
        # scroll_widget = Qw.QWidget()
        # scroll_widget.setLayout(grid)
        # scroll_area.setWidget(scroll_widget)
        # mainlayout.addWidget(scroll_area)
        # mainlayout.addSpacerItem(Qw.QSpacerItem(0, 0, Qw.QSizePolicy.Policy.Expanding))

        # Create a UI for tuning the parameters:
        # preset "slash"
        # slider for rotation, from -180 to 180 at 22.5 degree increments
        # slider for offset angle, from 0 to 360 at 45 degree increments
        # slider for offset factor, 0, 0.25 or 0.5
        # slider for scale factor, from 0.5 to 2 at 0.25 increments

        # Create a preview window for the current settings.
        preview = Qw.QWidget()
        preview_section = Qw.QHBoxLayout()
        preview_symbol = Qw.QLabel(preview)
        preview_symbol.setFixedSize(200, 200)
        preview_bar = Qw.QLabel(preview)
        preview_bar.setFixedSize(200, 200)
        # Stack the two previews on top of each other, so they overlap.
        preview_symbol.setGeometry(0, 0, 200, 200)
        preview_bar.setGeometry(0, 0, 200, 200)
        preview.setFixedSize(200, 200)
        preview_section.addWidget(preview)

        # settings area.
        settings_layout = Qw.QVBoxLayout()
        preview_section.addLayout(settings_layout)
        mainlayout.addLayout(preview_section)

        svg_data_vert = ut.load_symbol_svg(symbols["latex2e-OT1-_mid"], color)

        def render_preview():
            nonlocal svg_data_vert
            nonlocal negation
            bar_angle = negation.vert_angle
            bar_offset_angle = negation.offset_angle
            bar_offset_factor = negation.offset_factor
            bar_scale_factor = negation.scale_factor
            image_width = 200

            pixmap = Qg.QPixmap(image_width, image_width)
            pixmap.fill(Qc.Qt.GlobalColor.transparent)

            base_renderer = Qs.QSvgRenderer(svg_data_vert)
            base_renderer.setAspectRatioMode(Qc.Qt.AspectRatioMode.KeepAspectRatio)

            painter = Qg.QPainter(pixmap)
            painter.translate(image_width // 2, image_width // 2)
            painter.rotate(-bar_angle)

            midpoint_x = image_width // 2
            midpoint_y = image_width // 2

            # If scale < 1, we shrink the bar.
            if bar_scale_factor < 1:
                painter.scale(bar_scale_factor, bar_scale_factor)
                # Also offset it to keep the centerpoint in the same place.
                # painter.translate(
                #     (1 - bar_scale_factor) * image_width // 2,
                #     (1 - bar_scale_factor) * image_width // 2,
                # )

            painter.translate(-image_width // 2, -image_width // 2)

            if bar_offset_factor > 0:
                # Translate bar. This is a function of the image width to render at.
                # In the direction of the angle. Basically polar coordinates.
                bar_offset_angle -= bar_angle
                # print(bar_offset_angle)
                offset = bar_offset_factor * (image_width / 2) / min(bar_scale_factor, 1)
                painter.translate(
                    offset * np.cos(np.radians(bar_offset_angle)),
                    -offset * np.sin(np.radians(bar_offset_angle)),
                )
                midpoint_x += offset * np.cos(np.radians(bar_offset_angle)) // 2
                midpoint_y += -offset * np.sin(np.radians(bar_offset_angle)) // 2

            base_renderer.render(painter)
            # Draw a dot for the centerpoint
            painter.setPen(Qg.QPen(Qc.Qt.GlobalColor.red))
            # painter.drawEllipse(midpoint_x - 2, midpoint_y - 2, 4, 4)
            # print(midpoint_x, midpoint_y)

            preview_bar.setPixmap(pixmap)
            painter.end()

            pixmap2 = Qg.QPixmap(image_width, image_width)
            pixmap2.fill(Qc.Qt.GlobalColor.transparent)
            painter2 = Qg.QPainter(pixmap2)
            # Next, we render the symbol on top.
            # We just need to scale it down in case the scale factor is greater than 1.
            symbol_renderer = Qs.QSvgRenderer(svg_data)
            symbol_renderer.setAspectRatioMode(Qc.Qt.AspectRatioMode.KeepAspectRatio)
            if bar_scale_factor > 1:
                painter2.translate(image_width // 2, image_width // 2)
                painter2.scale(1 / bar_scale_factor, 1 / bar_scale_factor)
                painter2.translate(-image_width // 2, -image_width // 2)
            symbol_renderer.render(painter2)
            painter2.end()
            preview_symbol.setPixmap(pixmap2)
            update_sliders()

        def set_to_slash():
            nonlocal negation
            negation = st.Negation.from_string("slash")
            render_preview()

        def update_preview():
            nonlocal negation
            # Grab values from the sliders.
            angle = angle_slider.value() * 11.25
            offset_angle = offset_angle_slider.value() * 45
            if offset_factor_slider.value() < 4:
                offset_factor = offset_factor_slider.value() * 0.15
            else:
                offset_factor = 1.2
            scale_factor = scale_factor_slider.value() * 0.25
            negation = st.Negation(angle, offset_angle, offset_factor, scale_factor)
            render_preview()
            settings_label.setText(str(negation))

        def update_sliders():
            nonlocal negation
            nonlocal angle_slider
            nonlocal offset_angle_slider
            nonlocal offset_factor_slider
            nonlocal scale_factor_slider
            # print(f"setting sliders to {negation=}")
            angle_slider.setValue(int(negation.angle / 11.25))
            offset_angle_slider.setValue(int(negation.offset_angle / 45))
            if negation.offset_factor < 1.2:
                offset_factor_slider.setValue(int(negation.offset_factor / 0.15))
            else:
                offset_factor_slider.setValue(4)
            offset_factor_slider.setValue(int(negation.offset_factor / 0.15))
            scale_factor_slider.setValue(int(negation.scale_factor / 0.25))

        # Show a preview of the current settings as a string.
        settings_label = Qw.QLabel()
        settings_layout.addWidget(settings_label)

        # Controls:
        slash_button = Qw.QPushButton("Slash")
        slash_button.clicked.connect(set_to_slash)
        settings_layout.addWidget(slash_button)

        angle_slider = Qw.QSlider(Qc.Qt.Orientation.Horizontal)
        # We want to support 11.25 degree increments from 0 to 180.
        # That means we need 16 steps.
        angle_slider.setRange(0, 16)
        angle_slider.setTickInterval(1)
        angle_slider.setTickPosition(Qw.QSlider.TickPosition.TicksBelow)

        offset_angle_slider = Qw.QSlider(Qc.Qt.Orientation.Horizontal)
        # We want to support 45 degree increments from 0 to 360.
        # That means we need 8 steps.
        offset_angle_slider.setRange(0, 8)
        offset_angle_slider.setTickInterval(1)
        offset_angle_slider.setTickPosition(Qw.QSlider.TickPosition.TicksBelow)

        offset_factor_slider = Qw.QSlider(Qc.Qt.Orientation.Horizontal)
        offset_factor_slider.setRange(0, 4)
        offset_factor_slider.setTickInterval(1)
        offset_factor_slider.setTickPosition(Qw.QSlider.TickPosition.TicksBelow)

        scale_factor_slider = Qw.QSlider(Qc.Qt.Orientation.Horizontal)
        # We want to support values from 0.5 to 2 in 0.25 increments.
        # That means we need 7 steps.
        scale_factor_slider.setRange(2, 14)
        scale_factor_slider.setTickInterval(1)
        scale_factor_slider.setTickPosition(Qw.QSlider.TickPosition.TicksBelow)

        update_sliders()

        angle_slider.valueChanged.connect(update_preview)
        offset_angle_slider.valueChanged.connect(update_preview)
        offset_factor_slider.valueChanged.connect(update_preview)
        scale_factor_slider.valueChanged.connect(update_preview)

        label_angle = Qw.QLabel("Angle")
        settings_layout.addWidget(label_angle)
        settings_layout.addWidget(angle_slider)

        label_offset_angle = Qw.QLabel("Offset Angle")
        settings_layout.addWidget(label_offset_angle)
        settings_layout.addWidget(offset_angle_slider)

        label_offset_factor = Qw.QLabel("Offset Factor")
        settings_layout.addWidget(label_offset_factor)
        settings_layout.addWidget(offset_factor_slider)

        label_scale_factor = Qw.QLabel("Scale Factor")
        settings_layout.addWidget(label_scale_factor)
        settings_layout.addWidget(scale_factor_slider)

        update_preview()

        mainwindow.show()
        app.exec()
        # Print the settings
        print(f"{s_from} -/ {negation} /- {s_to}")


if __name__ == "__main__":
    main()
