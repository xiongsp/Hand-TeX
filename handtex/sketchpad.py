import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtWidgets as Qw
from PySide6.QtCore import Signal
from loguru import logger
from rdp import rdp

import handtex.structures as st
from handtex.constants import CANVAS_SIZE


class Sketchpad(Qw.QGraphicsView):
    """
    A drawing area for sketching symbols.
    """

    strokes: list[list[tuple[int, int]]]
    stroke_items: list[Qw.QGraphicsPathItem]
    current_stroke: list[tuple[int, int]] | None
    current_path: Qg.QPainterPath | None
    current_path_item: Qw.QGraphicsPathItem | None

    redo_strokes: list[list[tuple[int, int]]]
    redo_items: list[Qw.QGraphicsPathItem]

    can_undo = Signal(bool)
    can_redo = Signal(bool)
    new_drawing = Signal()

    pen_width: int

    overlay: Qw.QWidget

    def __init__(self, parent=None):
        super().__init__(parent)
        scene = Qw.QGraphicsScene(self)
        scene.setSceneRect(0, 0, 100, 100)  # Set a fixed scene size
        self.setScene(scene)
        self.strokes = []
        self.stroke_items = []  # QGraphicsPathItems for undo/redo actions
        self.current_stroke = None
        self.current_path = None
        self.current_path_item = None

        self.set_up_overlay()

        self.pen_width = 10

        pen_color = self.palette().color(Qg.QPalette.Text)
        self.pen = Qg.QPen(pen_color, self.pen_width)
        self.pen.setCapStyle(Qc.Qt.RoundCap)
        self.pen.setJoinStyle(Qc.Qt.RoundJoin)

        self.redo_strokes = []
        self.redo_items = []

        self.setMouseTracking(True)  # Enable mouse tracking to get accurate positions
        self.setSceneRect(0, 0, self.width(), self.height())

    def set_up_overlay(self):
        # Create overlay container
        self.overlay = Qw.QWidget(self)
        self.overlay.setAttribute(Qc.Qt.WA_TransparentForMouseEvents, True)  # Click-through

        # Layout for stacking icon and label.
        layout = Qw.QVBoxLayout()
        layout.setAlignment(Qc.Qt.AlignCenter)

        text_label = Qw.QLabel("Draw here")
        font = text_label.font()
        font.setPointSize(int(3 * font.pointSize()))
        text_label.setFont(font)
        text_label.setAlignment(Qc.Qt.AlignCenter)

        icon_label = Qw.QLabel()
        icon_label.setObjectName("overlay-icon")
        icon = Qg.QIcon.fromTheme("draw-freehand")
        pixmap = icon.pixmap(128, 128)  # Set icon size
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qc.Qt.AlignCenter)

        layout.addWidget(text_label)
        layout.addWidget(icon_label)
        self.overlay.setLayout(layout)

        self.updateOverlayPosition()

    def updateOverlayPosition(self):
        """
        Keep overlay centered in the view.
        """
        self.overlay.setGeometry(0, 0, self.viewport().width(), self.viewport().height())

    def resizeEvent(self, event: Qg.QResizeEvent) -> None:
        """
        Adjust the scene size when the widget is resized.
        """
        self.setSceneRect(0, 0, self.width(), self.height())
        self.updateOverlayPosition()
        super().resizeEvent(event)

    def wheelEvent(self, event: Qg.QWheelEvent) -> None:
        """
        Ignore mouse wheel events to prevent scrolling.
        """
        event.ignore()

    def recolor(self) -> None:
        """
        Recolor the pen to match the text color.
        Change the greeter icon's color.
        """
        pen_color = self.palette().color(Qg.QPalette.Text)
        self.pen.setColor(pen_color)
        for item in self.stroke_items:
            item.setPen(self.pen)

        # Recolor the overlay icon.
        icon = Qg.QIcon.fromTheme("draw-freehand")
        pixmap = icon.pixmap(128, 128)
        icon_label = self.overlay.findChild(Qw.QLabel, "overlay-icon")
        icon_label.setPixmap(pixmap)

    def mousePressEvent(self, event: Qg.QMouseEvent) -> None:
        if event.button() == Qc.Qt.LeftButton:
            self.overlay.hide()
            # Start a new stroke.
            self.current_stroke = []
            self.current_path = Qg.QPainterPath()
            pos = self.mapToScene(event.pos())
            # Clear the redo stack when starting a new stroke.
            self.redo_strokes.clear()
            self.redo_items.clear()
            self.current_stroke.append((round(pos.x()), round(pos.y())))
            self.current_path.moveTo(pos)
            self.current_path_item = self.scene().addPath(self.current_path, self.pen)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: Qg.QMouseEvent) -> None:
        if self.current_stroke is not None:
            # Continue the stroke.
            pos = self.mapToScene(event.pos())
            self.current_stroke.append((round(pos.x()), round(pos.y())))
            self.current_path.lineTo(pos)
            self.current_path_item.setPath(self.current_path)
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: Qg.QMouseEvent) -> None:
        if event.button() == Qc.Qt.LeftButton and self.current_stroke:
            # First, purge any consecutive duplicate points.
            stroke = [self.current_stroke[0]]
            for point in self.current_stroke[1:]:
                if point != stroke[-1]:
                    stroke.append(point)
            self.current_stroke = stroke

            # Finish the stroke.
            if len(self.current_stroke) == 1:
                # If the stroke has a single point, add 3 more to form a little square.
                pos = Qc.QPoint(*self.current_stroke[0])
                self.current_path.lineTo(pos + Qc.QPoint(1, 0))
                self.current_path.lineTo(pos + Qc.QPoint(1, 1))
                self.current_path.lineTo(pos + Qc.QPoint(0, 1))
                # self.current_stroke.append((pos.x(), pos.y()))
                self.current_path_item.setPath(self.current_path)

            self.strokes.append(self.current_stroke)
            self.stroke_items.append(self.current_path_item)
            self.current_stroke = None
            self.current_path_item = None
            event.accept()
            self.new_drawing.emit()
        else:
            event.ignore()

        self.can_undo.emit(bool(self.strokes))
        self.can_redo.emit(bool(self.redo_strokes))

    def clear(self) -> None:
        """
        Clear the sketchpad.
        """
        self.scene().clear()
        self.strokes.clear()
        self.stroke_items.clear()
        self.redo_strokes.clear()
        self.redo_items.clear()
        self.can_undo.emit(False)
        self.can_redo.emit(False)

    def is_empty(self) -> bool:
        """
        Check if the sketchpad is empty.
        """
        return not self.strokes

    def undo(self):
        """
        Undo the last stroke.
        """
        if self.strokes:
            stroke = self.strokes.pop()
            item = self.stroke_items.pop()
            self.scene().removeItem(item)
            self.redo_strokes.append(stroke)
            self.redo_items.append(item)
        self.can_undo.emit(bool(self.strokes))
        self.can_redo.emit(bool(self.redo_strokes))
        if self.strokes:
            self.new_drawing.emit()

    def redo(self):
        """
        Redo the last undone stroke.
        """
        if self.redo_strokes:
            stroke = self.redo_strokes.pop()
            item = self.redo_items.pop()
            self.strokes.append(stroke)
            self.stroke_items.append(item)
            self.scene().addItem(item)
            self.new_drawing.emit()
        self.can_undo.emit(bool(self.strokes))
        self.can_redo.emit(bool(self.redo_strokes))

    def get_clean_strokes(
        self, simplify: bool = False
    ) -> tuple[list[list[tuple[int, int]]], float, int, int]:
        """
        Normalize the coordinates to a 0-CANVAS_SIZE space.
        Apply the RDP algorithm to limit the number of points in each stroke.
        """
        strokes = self.strokes
        if not strokes:
            return [], 1, 0, 0
        clean_strokes = purge_duplicate_strokes(strokes)

        if simplify:
            clean_strokes = [simplify_stroke(stroke) for stroke in clean_strokes]

        centered_clean_strokes, scale, x_offset, y_offset = rescale_and_center_viewport(
            clean_strokes, self.sceneRect().width(), self.sceneRect().height()
        )
        return centered_clean_strokes, scale, x_offset, y_offset

    def load_strokes(self, drawing: st.SymbolDrawing) -> None:
        """
        Load strokes into the sketchpad, as if they had been drawn.
        This requires undoing the transformations applied to the strokes.

        :param drawing: The drawing to load.
        """

        logger.debug(f"Loading {len(drawing.strokes)} strokes into the sketchpad.")
        self.clear()
        self.strokes.clear()
        for stroke in drawing.strokes:
            path = Qg.QPainterPath()
            stroke = [
                (
                    round((x - drawing.x_offset) / drawing.scaling),
                    round((y - drawing.y_offset) / drawing.scaling),
                )
                for x, y in stroke
            ]
            self.strokes.append(stroke)
            path.moveTo(*stroke[0])
            for point in stroke[1:]:
                path.lineTo(*point)
            # Handle single point strokes.
            if len(stroke) == 1:
                x, y = stroke[0]
                path.lineTo(x + 1, y + 1)
            item = self.scene().addPath(path, self.pen)
            self.stroke_items.append(item)
        self.can_undo.emit(bool(self.strokes))

    def set_pen_width(self, width: int) -> None:
        """
        Set the pen width.
        """
        self.pen_width = width
        self.pen.setWidth(width)
        for item in self.stroke_items:
            item.setPen(self.pen)


def purge_duplicate_strokes(strokes: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    cleaned_strokes = []
    # Check for identical strokes.
    for stroke in strokes:
        if stroke in cleaned_strokes:
            continue
        cleaned_strokes.append(stroke)
    return cleaned_strokes


def rescale_and_center_viewport(
    coordinates: list[list[tuple[int, int]]], viewport_width: float, viewport_height: float
) -> tuple[list[list[tuple[int, int]]], float, int, int]:
    """
    Transform the coordinate space from the viewport to a 0-CANVAS_SIZE space, and then scale and center the coordinates.
    Preserve the aspect ratio of the viewport.

    :param coordinates: The coordinates to transform.
    :param viewport_width: The width of the viewport.
    :param viewport_height: The height of the viewport.
    :return: The transformed coordinates, the scaling factor, and the x and y offset.
    """
    # Step 1: Rescale the viewport
    initial_scale = CANVAS_SIZE / max(viewport_width, viewport_height)

    scaled_coordinates = []
    for sublist in coordinates:
        scaled_sublist = [(x * initial_scale, y * initial_scale) for x, y in sublist]
        scaled_coordinates.append(scaled_sublist)

    # Step 2: Scale and center the rescaled coordinates
    all_points = [point for sublist in scaled_coordinates for point in sublist]

    if len(all_points) == 1:
        return (
            [[(CANVAS_SIZE // 2, CANVAS_SIZE // 2)]],
            1,
            CANVAS_SIZE / 2 - all_points[0][0],
            CANVAS_SIZE / 2 - all_points[0][1],
        )

    xs, ys = zip(*all_points)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width, height = max_x - min_x, max_y - min_y

    # Determine the scaling factor to fit in CANVAS_SIZExCANVAS_SIZE, preserving aspect ratio
    scale_factor = CANVAS_SIZE / max(width, height)

    # Correct the scaling factor if needed
    scale_factor = scale_correction_function(scale_factor)

    offset_x = round((CANVAS_SIZE - width * scale_factor) / 2 - min_x * scale_factor)
    offset_y = round((CANVAS_SIZE - height * scale_factor) / 2 - min_y * scale_factor)

    # Scale and center all coordinates
    centered_coords = []
    for sublist in scaled_coordinates:
        centered_sublist = [
            (int(x * scale_factor + offset_x), int(y * scale_factor + offset_y)) for x, y in sublist
        ]
        centered_coords.append(centered_sublist)

    return centered_coords, initial_scale * scale_factor, offset_x, offset_y


def scale_correction_function(x: float) -> float:
    r"""
    Fully scale up until a factor of x=4, then cap the max scaling at y=5 for x>8.
    This decays asymptotically to y=2 with an inflection point at x=20.

         x < 4 : x
    4 <= x < 8 : -\frac{1}{e^{\left(x-4\right)}}+5
    8 <= x     : 2.0+0.0614754+3\left(1-\frac{1}{1+e^{-0.3\left(x-20\right)}}\right)\right\}

    https://www.desmos.com/calculator/v3iimkbieg

    :param x: The scaling factor.
    :return: The corrected scaling factor.
    """
    if x < 4:
        return x
    elif x < 8:
        return -1 / (2.71828 ** (x - 4)) + 5
    else:
        return 2.0614754 + 3 * (1 - 1 / (1 + 2.71828 ** (-0.3 * (x - 20))))


def simplify_stroke(stroke, epsilon=1.25):
    """
    Simplify a stroke using the Ramer-Douglas-Peucker algorithm.
    Used on training data to keep the database small.
    The algorithm is too slow for real-time use, it's faster to
    just draw a bunch of tiny strokes.

    :param stroke: List of points [[x1, y1], [x2, y2], ...]
    :param epsilon: Tolerance level for simplification.
    :return: Simplified stroke.
    """
    if len(stroke) < 3:
        return stroke  # Not enough points to simplify
    return rdp(stroke, epsilon=epsilon)
