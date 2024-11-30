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

        self.pen_width = 10

        pen_color = self.palette().color(Qg.QPalette.Text)
        self.pen = Qg.QPen(pen_color, self.pen_width)
        self.pen.setCapStyle(Qc.Qt.RoundCap)
        self.pen.setJoinStyle(Qc.Qt.RoundJoin)

        self.redo_strokes = []
        self.redo_items = []

        self.setMouseTracking(True)  # Enable mouse tracking to get accurate positions
        self.setSceneRect(0, 0, self.width(), self.height())

    def resizeEvent(self, event: Qg.QResizeEvent) -> None:
        """
        Adjust the scene size when the widget is resized.
        """
        self.setSceneRect(0, 0, self.width(), self.height())
        super().resizeEvent(event)

    def wheelEvent(self, event: Qg.QWheelEvent) -> None:
        """
        Ignore mouse wheel events to prevent scrolling.
        """
        event.ignore()

    def recolor_pen(self) -> None:
        """
        Recolor the pen to match the text color.
        """
        pen_color = self.palette().color(Qg.QPalette.Text)
        self.pen.setColor(pen_color)
        for item in self.stroke_items:
            item.setPen(self.pen)

    def mousePressEvent(self, event: Qg.QMouseEvent) -> None:
        if event.button() == Qc.Qt.LeftButton:
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
                # If the stroke has only one point, add a second point right next to it.
                pos = Qc.QPoint(*self.current_stroke[0])
                pos += Qc.QPoint(1, 1)
                # self.current_stroke.append((pos.x(), pos.y()))
                self.current_path.lineTo(pos)
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

    def get_clean_strokes(self) -> tuple[list[list[tuple[int, int]]], float, int, int]:
        """
        Normalize the coordinates to a 0-CANVAS_SIZE space.
        Apply the RDP algorithm to limit the number of points in each stroke.
        """
        strokes = self.strokes
        if not strokes:
            return [], 1, 0, 0
        clean_strokes = purge_duplicate_strokes(strokes)
        clean_strokes, scale1 = rescale_viewport(clean_strokes, self.sceneRect())
        clean_strokes, scale2, x_offset, y_offset = scale_and_center(clean_strokes)
        return clean_strokes, scale1 * scale2, x_offset, y_offset

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


def rescale_viewport(
    coordinates: list[list[tuple[int, int]]], rect: Qc.QRectF
) -> tuple[list[list[tuple[int, int]]], float]:
    """
    Transform the coordinate space from the viewport to a 0-CANVAS_SIZE space.
    Preserve the aspect ratio of the viewport.

    :param coordinates: The coordinates to transform.
    :param rect: The viewport rectangle.
    :return: The transformed coordinates and the scaling factor.
    """
    width, height = rect.width(), rect.height()

    scale = CANVAS_SIZE / max(width, height)

    scaled_coordinates = []
    for sublist in coordinates:
        scaled_sublist = [(int(x * scale), int(y * scale)) for x, y in sublist]
        scaled_coordinates.append(scaled_sublist)
    return scaled_coordinates, scale


def scale_and_center(
    coordinates: list[list[tuple[int, int]]]
) -> tuple[list[list[tuple[int, int]]], float, int, int]:
    all_points = [point for sublist in coordinates for point in sublist]
    """
    Scale and center the coordinates in a CANVAS_SIZExCANVAS_SIZE space.
    
    :param coordinates: The coordinates to scale and center.
    :return: The scaled and centered coordinates, the scaling factor, and the x and
             y offset.
    """

    # Handle the case where there is only a single point
    if len(all_points) == 1:
        return (
            [[(CANVAS_SIZE / 2, CANVAS_SIZE / 2)]],
            1,
            CANVAS_SIZE / 2 - all_points[0][0],
            CANVAS_SIZE / 2 - all_points[0][1],
        )  # Center the single point in the CANVAS_SIZExCANVAS_SIZE space

    xs, ys = zip(*all_points)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width, height = max_x - min_x, max_y - min_y

    # Determine the scaling factor to fit in CANVAS_SIZExCANVAS_SIZE, preserving aspect ratio
    scale_factor = CANVAS_SIZE / max(width, height)

    scale_factor = scale_correction_function(scale_factor)

    offset_x = (CANVAS_SIZE - width * scale_factor) / 2 - min_x * scale_factor
    offset_y = (CANVAS_SIZE - height * scale_factor) / 2 - min_y * scale_factor

    # Scale and center all coordinates.
    scaled_coordinates = []
    for sublist in coordinates:
        scaled_sublist = [
            (int(x * scale_factor + offset_x), int(y * scale_factor + offset_y)) for x, y in sublist
        ]
        scaled_coordinates.append(scaled_sublist)

    return scaled_coordinates, scale_factor, offset_x, offset_y


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
