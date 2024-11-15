from enum import IntEnum

import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtSvg as Qs
import PySide6.QtWidgets as Qw
from PySide6.QtCore import Qt

import handtex.state_saver as ss
import handtex.structures as st
import handtex.utils as ut
from handtex.ui_generated_files.ui_SymbolList import Ui_SymbolList


class SearchMode(IntEnum):
    COMMAND = 0
    ID = 1
    SIMILAR = 2
    UNIQUE = 3


class SymbolList(Qw.QWidget, Ui_SymbolList):

    symbols: dict[str, st.Symbol]
    similar_symbols: dict[str, set[str]]
    icon_size: int
    pixmap_cache: dict[str, Qg.QPixmap]
    current_symbol_keys: list[str | None]

    state_saver: ss.StateSaver

    def __init__(
        self,
        symbols: dict[str, st.Symbol],
        similar_symbols: dict[str, set[str]],
        parent=None,
    ):
        super(SymbolList, self).__init__(parent)
        self.setupUi(self)
        self.symbols = symbols
        self.similar_symbols = similar_symbols

        self.icon_size = 100
        self.listWidget.setIconSize(Qc.QSize(self.icon_size, self.icon_size))

        self.pixmap_cache = {}
        self.current_symbol_keys = list(symbols.keys())

        self.show_symbols()

        self.listWidget.currentItemChanged.connect(self.on_symbol_selected)
        self.listWidget.setCurrentRow(0)

        self.comboBox_search_mode.currentIndexChanged.connect(self.listWidget.clearSelection())
        self.comboBox_search_mode.currentIndexChanged.connect(self.search_symbols)
        self.lineEdit_search.textEdited.connect(self.search_symbols)

        # Init UI.
        font = self.label_id.font()
        font.setPointSize(int(1.5 * font.pointSize()))
        self.label_id.setFont(font)
        # Make the right splitter side 250px wide.
        self.splitter.setSizes([self.width() - 320, 320])
        self.listWidget.verticalScrollBar().setSingleStep(40)

        self.state_saver = ss.StateSaver("symbol_list")
        self.init_state_saver()
        self.state_saver.restore()

    def init_state_saver(self) -> None:
        """
        Load the state from the state saver.
        """
        self.state_saver.register(
            self,
            self.splitter,
            self.comboBox_search_mode,
        )

    def closeEvent(self, event) -> None:
        self.state_saver.save()
        event.accept()

    def search_symbols(self) -> None:
        """
        Check the mode and current search text, then update the list of symbols.
        """
        search_text = self.lineEdit_search.text().strip()
        search_mode = SearchMode(self.comboBox_search_mode.currentIndex())

        if search_mode == SearchMode.COMMAND:
            self.current_symbol_keys = [
                key
                for key in self.symbols
                if search_text.lower() in self.symbols[key].command.lower()
            ]
        elif search_mode == SearchMode.ID:
            self.current_symbol_keys = [
                key for key in self.symbols if search_text.lower() in key.lower()
            ]
        elif search_mode == SearchMode.SIMILAR:
            if search_text in self.similar_symbols:
                # List all symbols that are in the similarity group.
                self.current_symbol_keys = list(self.similar_symbols[search_text])
            else:
                # List all symbols that are in similarity groups, grouped
                # by being in a similarity group.
                self.current_symbol_keys = []
                for key in self.similar_symbols:
                    if key in self.current_symbol_keys:
                        continue
                    if not search_text or search_text.lower() in key.lower():
                        self.current_symbol_keys.append(key)
                        self.current_symbol_keys.extend(list(self.similar_symbols[key]))
                        self.current_symbol_keys.append(None)
        elif search_mode == SearchMode.UNIQUE:
            # Exclude all symbols that are in similarity groups.
            self.current_symbol_keys = [
                key
                for key in self.symbols
                if search_text.lower() in key.lower() and key not in self.similar_symbols
            ]

        self.show_symbols()

    def on_symbol_selected(self, *_) -> None:
        current_index = self.listWidget.currentRow()
        if current_index < 0:
            return
        if self.listWidget.count() == 0:
            return
        if current_index >= len(self.current_symbol_keys):
            return

        symbol_key = self.current_symbol_keys[current_index]
        if symbol_key is None:
            return
        self.show_symbol_details(symbol_key)

    def show_symbols(self) -> None:
        self.listWidget.clear()
        separator_pixmap = Qg.QPixmap(self.icon_size, self.icon_size)
        separator_pixmap.fill(Qt.transparent)
        # Draw a vertical line in text color.
        painter = Qg.QPainter(separator_pixmap)
        painter.setPen(self.palette().color(Qg.QPalette.Text))
        painter.drawLine(
            separator_pixmap.width() // 2,
            int(separator_pixmap.height() * 0.2),
            separator_pixmap.width() // 2,
            int(separator_pixmap.height() * 0.8),
        )
        painter.end()
        for key in self.current_symbol_keys:
            if key is None:
                # Add a spacer item to force a new line (like a line break).
                spacer_item = Qw.QListWidgetItem(separator_pixmap, " ")
                spacer_item.setFlags(spacer_item.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsEnabled)
                # Set a size hint wide enough to take up the rest of the row.
                # Adjust width value based on how wide your items are.
                self.listWidget.addItem(spacer_item)
                continue
            label = self.symbols[key].command
            pixmap = self.get_symbol_pixmap(key)

            item = Qw.QListWidgetItem(pixmap, label)
            self.listWidget.addItem(item)

        self.label_count.setText(str(len(self.current_symbol_keys)))

    def show_symbol_details(self, symbol_key: str):
        symbol = self.symbols[symbol_key]
        self.label_id.setText(symbol.key)
        self.label_command.setText(symbol.command)
        self.label_mode.setText(symbol.mode_str())
        self.label_package.setVisible(not symbol.package_is_default())
        self.label_package_label.setVisible(not symbol.package_is_default())
        self.label_package.setText(symbol.package)
        self.label_fontenc.setVisible(not symbol.fontenc_is_default())
        self.label_fontenc_label.setVisible(not symbol.fontenc_is_default())
        self.label_fontenc.setText(symbol.fontenc)

        if symbol_key in self.similar_symbols:
            self.label_similar.setText(", ".join(self.similar_symbols[symbol_key]))
        else:
            self.label_similar.setText("")

        hex_color = self.palette().color(Qg.QPalette.Text).name()
        self.widget_symbol_view.load(ut.load_symbol_svg(symbol, hex_color))
        self.widget_symbol_view.renderer().setAspectRatioMode(Qc.Qt.KeepAspectRatio)
        self.widget_symbol_view.setFixedSize(200, 200)

    def get_symbol_pixmap(self, symbol_key: str) -> Qg.QPixmap:
        if symbol_key in self.pixmap_cache:
            return self.pixmap_cache[symbol_key]
        hex_color = self.palette().color(Qg.QPalette.Text).name()
        svg_xml = ut.load_symbol_svg(self.symbols[symbol_key], hex_color)
        svg_renderer = Qs.QSvgRenderer()
        svg_renderer.load(svg_xml)
        svg_renderer.setAspectRatioMode(Qt.KeepAspectRatio)
        pixmap = Qg.QPixmap(self.icon_size, self.icon_size)
        pixmap.fill(Qt.transparent)
        svg_renderer.render(Qg.QPainter(pixmap))
        self.pixmap_cache[symbol_key] = pixmap
        return pixmap
