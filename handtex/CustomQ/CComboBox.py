from typing import Any

import PySide6.QtWidgets as Qw


class CComboBox(Qw.QComboBox):
    """
    Extends the functionality with custom helpers
    And includes a secondary array for data linked to each item
    """

    def __init__(self, parent=None) -> None:
        Qw.QComboBox.__init__(self, parent)
        self._linked_data = []

    def clear(self) -> None:
        Qw.QComboBox.clear(self)
        self._linked_data.clear()

    def addTextItemLinkedData(self, text: str, data: Any) -> None:
        self.addItem(text)
        self._linked_data.append(data)

    def setCurrentIndexByLinkedData(self, data: Any) -> None:
        self.setCurrentIndex(self._linked_data.index(data))

    def indexLinkedData(self, data: Any) -> int:
        return self._linked_data.index(data)

    def currentLinkedData(self) -> None:
        if self.currentIndex() == -1:
            return None
        return self._linked_data[self.currentIndex()]
