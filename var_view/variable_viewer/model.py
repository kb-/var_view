# var_view/variable_viewer/model.py

from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt
import logging

logger = logging.getLogger(__name__)


class VariableStandardItemModel(QStandardItemModel):
    """
    Custom QStandardItemModel that supports dragging the variable path from the tree.
    """

    def __init__(self, viewer, alias="data_source", parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.alias = alias

    def supportedDragActions(self):
        return Qt.DropAction.CopyAction

    def mimeData(self, indexes):
        mime_data = super().mimeData(indexes)
        paths = []
        for idx in indexes:
            if idx.column() != 0:
                continue
            item = self.itemFromIndex(idx)
            if item:
                path = self.viewer.resolve_item_path(item)
                if path and not path.startswith(self.alias):
                    path = f"{self.alias}.{path}"
                paths.append(path)
        if paths:
            mime_data.setText("\n".join(paths))
            logger.debug(f"Dragging variable paths: {paths}")
        return mime_data

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        if index.column() == 0:
            return super().flags(index) | Qt.ItemFlag.ItemIsDragEnabled
        return super().flags(index)
