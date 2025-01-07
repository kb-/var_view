from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtCore import Qt, QModelIndex, QMimeData
import logging
from typing import List

logger = logging.getLogger(__name__)


class VariableStandardItemModel(QStandardItemModel):
    """
    A custom QStandardItemModel that supports dragging variable paths from a tree view.

    Attributes:
        viewer: The parent viewer that provides methods like `resolve_item_path`.
        alias: A string prefix added to variable paths during drag operations.
    """

    def __init__(self, viewer, alias: str = "data_source", parent=None):
        """
        Initializes the custom model.

        Args:
            viewer: The parent viewer object.
            alias: The prefix to use for variable paths during drag operations.
            parent: The parent object for the model, typically a QObject or None.
        """
        super().__init__(parent)
        self.viewer = viewer
        self.alias = alias

    def supportedDragActions(self) -> Qt.DropAction:
        """
        Returns the supported drag actions for this model.

        Returns:
            Qt.DropAction.CopyAction: Indicates that drag-and-drop operations support copying.
        """
        return Qt.DropAction.CopyAction

    def mimeData(self, indexes: List[QModelIndex]) -> QMimeData:
        """
        Creates MIME data for drag-and-drop operations.

        Args:
            indexes: A list of QModelIndex objects representing the items being dragged.

        Returns:
            QMimeData: The MIME data containing the concatenated variable paths.
        """
        mime_data = super().mimeData(indexes)
        paths = []

        for idx in indexes:
            if idx.column() != 0:  # Only process column 0 items
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

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """
        Returns the item flags for a given index, including drag capability.

        Args:
            index: The QModelIndex to retrieve flags for.

        Returns:
            Qt.ItemFlags: The item flags, including Qt.ItemFlag.ItemIsDragEnabled if valid.
        """
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return super().flags(index) | Qt.ItemFlag.ItemIsDragEnabled
