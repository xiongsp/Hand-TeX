import PySide6.QtWidgets as Qw
import PySide6.QtCore as Qc
import PySide6.QtGui as Qg


class CScrollArea(Qw.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create a small circular tool button for scrolling to top
        self.scroll_to_top_btn = Qw.QToolButton(self.viewport())
        self.scroll_to_top_btn.setToolTip("Scroll to top")

        # Use the system's standard arrow-up icon
        self.scroll_to_top_btn.setIcon(self.style().standardIcon(Qw.QStyle.SP_ArrowUp))

        # Make it a small circle: 30×30 px
        self.scroll_to_top_btn.setFixedSize(30, 30)

        # Initial stylesheet using the palette highlight color
        self.update_scroll_to_top_btn_style()

        # Hide initially
        self.scroll_to_top_btn.hide()

        # When clicked, scroll to top
        self.scroll_to_top_btn.clicked.connect(self.scroll_to_top)

        # Track vertical scrollbar movement to decide when to show/hide
        self.verticalScrollBar().valueChanged.connect(self.handle_scroll_value_changed)

    def handle_scroll_value_changed(self, value: int) -> None:
        """Show the scroll-to-top button if scrolled >20 px, else hide it."""
        if value > 20:
            self.scroll_to_top_btn.show()
            self.scroll_to_top_btn.raise_()
        else:
            self.scroll_to_top_btn.hide()

    def resizeEvent(self, event: Qg.QResizeEvent) -> None:
        """Position the button near the top after any resizing."""
        super().resizeEvent(event)

        # Center the button horizontally at the top of the viewport
        x = (self.viewport().width() - self.scroll_to_top_btn.width()) // 2
        y = 10  # e.g. 10 px from the top
        self.scroll_to_top_btn.move(x, y)

    def scroll_to_top(self) -> None:
        """Scroll all the way back to the top."""
        self.verticalScrollBar().setValue(0)

    def changeEvent(self, event) -> None:
        """
        Listen for palette change events to update the scroll-to-top button's style.
        """
        super().changeEvent(event)
        if event.type() == Qc.QEvent.ApplicationPaletteChange:
            # Re-apply styling when the palette changes
            self.update_scroll_to_top_btn_style()

    def update_scroll_to_top_btn_style(self) -> None:
        """
        Update the scroll-to-top button's stylesheet to reflect
        the current palette highlight color and other palette properties.
        """
        # Get the highlight color from the current palette
        highlight_color = self.palette().highlight().color().name(Qg.QColor.HexArgb)

        self.scroll_to_top_btn.setStyleSheet(
            f"""
            QToolButton {{
                background-color: {highlight_color};
                border: none;
                border-radius: 15px; /* half of fixedSize for a circle */
            }}
        """
        )
