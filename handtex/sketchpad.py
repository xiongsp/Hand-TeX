import PySide6.QtWidgets as Qw
import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
from PySide6.QtCore import Signal

from loguru import logger
from rdp import rdp


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

        self.pen_width = 20

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
                self.current_stroke.append((pos.x(), pos.y()))
                self.current_path.lineTo(pos)
                self.current_path_item.setPath(self.current_path)

            self.strokes.append(self.current_stroke)
            self.stroke_items.append(self.current_path_item)
            self.current_stroke = None
            self.current_path_item = None
            event.accept()
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
        self.can_undo.emit(bool(self.strokes))
        self.can_redo.emit(bool(self.redo_strokes))

    def clean_strokes(self) -> list[list[tuple[int, int]]]:
        """
        Normalize the coordinates to a 0-1000 space.
        Apply the RDP algorithm to limit the number of points in each stroke.
        """
        strokes = self.strokes
        clean_strokes = purge_duplicate_strokes(strokes)
        clean_strokes = scale_and_center(clean_strokes)
        clean_strokes = [simplify_stroke(stroke) for stroke in clean_strokes]
        return clean_strokes


def purge_duplicate_strokes(strokes: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    cleaned_strokes = []
    # Check for identical strokes.
    for stroke in strokes:
        if stroke in cleaned_strokes:
            continue
        cleaned_strokes.append(stroke)
    return cleaned_strokes


def scale_and_center(coordinates: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    all_points = [point for sublist in coordinates for point in sublist]

    # Handle the case where there is only a single point
    if len(all_points) == 1:
        return [[(500, 500)]]  # Center the single point in the 1000x1000 space

    xs, ys = zip(*all_points)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width, height = max_x - min_x, max_y - min_y

    # Determine the scaling factor to fit in 1000x1000, preserving aspect ratio.
    scale_factor = 1000 / max(width, height)

    offset_x = (1000 - width * scale_factor) / 2 - min_x * scale_factor
    offset_y = (1000 - height * scale_factor) / 2 - min_y * scale_factor

    # Scale and center all coordinates.
    scaled_coordinates = []
    for sublist in coordinates:
        scaled_sublist = [
            (int(x * scale_factor + offset_x), int(y * scale_factor + offset_y)) for x, y in sublist
        ]
        scaled_coordinates.append(scaled_sublist)

    return scaled_coordinates


def simplify_stroke(stroke, epsilon=1.25):
    """
    Simplify a stroke using the Ramer-Douglas-Peucker algorithm.
    :param stroke: List of points [[x1, y1], [x2, y2], ...]
    :param epsilon: Tolerance level for simplification.
    :return: Simplified stroke.
    """
    if len(stroke) < 3:
        return stroke  # Not enough points to simplify
    return rdp(stroke, epsilon=epsilon)
