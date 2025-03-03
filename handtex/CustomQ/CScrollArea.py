import PySide6.QtWidgets as Qw
import PySide6.QtCore as Qc
import PySide6.QtGui as Qg


class CScrollArea(Qw.QScrollArea):
    """
    A custom subclass for extra functionality.
    Current features:
    - A scroll-to-top button that appears when scrolled down.
    """

    scroll_to_top_icon_size = 32
    scroll_to_top_margin = 10
    scroll_to_top_scroll_margin = 20

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create the scroll-to-top button.
        self.scroll_to_top_btn = Qw.QToolButton(self.viewport())
        self.scroll_to_top_btn.setToolTip("Scroll to top")
        self.scroll_to_top_btn.setIcon(Qg.QIcon.fromTheme("go-up"))
        self.scroll_to_top_btn.setFixedSize(
            self.scroll_to_top_icon_size, self.scroll_to_top_icon_size
        )
        self.update_scroll_to_top_btn_style()
        self.scroll_to_top_btn.hide()
        self.scroll_to_top_btn.clicked.connect(self.scroll_to_top)
        # Track vertical scrollbar movement to decide when to show/hide.
        self.verticalScrollBar().valueChanged.connect(self.handle_scroll_value_changed)

    def handle_scroll_value_changed(self, value: int) -> None:
        if value > self.scroll_to_top_scroll_margin:
            self.scroll_to_top_btn.show()
            self.scroll_to_top_btn.raise_()  # Keep it above the viewport.
        else:
            self.scroll_to_top_btn.hide()

    def resizeEvent(self, event: Qg.QResizeEvent) -> None:
        """
        Position the button near the top after any resizing.
        """
        super().resizeEvent(event)

        # Center the button horizontally at the top of the viewport.
        x = (self.viewport().width() - self.scroll_to_top_btn.width()) // 2
        y = self.scroll_to_top_margin
        self.scroll_to_top_btn.move(x, y)

    def scroll_to_top(self) -> None:
        """
        Scroll all the way back to the top.
        """
        self.verticalScrollBar().setValue(0)

    def changeEvent(self, event) -> None:
        """
        Listen for palette change events to update the scroll-to-top button's style.
        """
        super().changeEvent(event)
        if event.type() == Qc.QEvent.PaletteChange:
            # Re-apply styling when the palette changes
            # Do a delay timer to allow the palette to update first.
            Qc.QTimer.singleShot(1, self.update_scroll_to_top_btn_style)

    def update_scroll_to_top_btn_style(self) -> None:
        """
        Update the scroll-to-top button's stylesheet to reflect
        the current palette highlight color and other palette properties.
        """

        self.scroll_to_top_btn.setStyleSheet(
            f"""
            QToolButton {{
                background-color: palette(Highlight);
                border: none;
                border-radius: {self.scroll_to_top_icon_size // 2}px;
            }}
        """
        )
