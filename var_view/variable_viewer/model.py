from PyQt6.QtGui import QStandardItemModel, QStandardItem
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
        from PyQt6.QtCore import QMimeData
        mime_data = QMimeData()
        all_lines = []

        for idx in indexes:
            if idx.column() != 0:
                continue
            item = self.itemFromIndex(idx)
            if item:
                # Prefer the precomputed safe drag text (UserRole+3)
                safe_path = item.data(Qt.ItemDataRole.UserRole + 3)
                if safe_path:
                    all_lines.append(safe_path)
                else:
                    # Fallback to computing the multiline path if needed
                    lines = self.compute_multiline_path_for_item(item)
                    all_lines.extend(lines)

        if all_lines:
            mime_data.setText("\n".join(all_lines))
            logger.debug(f"Dragging multiline variable paths:\n{mime_data.text()}")

        return mime_data

    def compute_multiline_path_for_item(self, item: QStandardItem) -> list:
        try:
            lines = []
            ancestors = []
            current = item
            while current is not None:
                ancestors.append(current)
                current = current.parent()
            ancestors.reverse()

            # Fallback for items without a complex ancestry:
            if len(ancestors) == 1:
                path = self.viewer.resolve_item_path(item)
                if path and not (path.startswith(f"{self.alias}.") or path.startswith(
                        f"{self.alias}[")):
                    path = f"{self.alias}.{path}"
                return [path]

            current_expr = self.alias
            tmp_var_count = 0

            for node in ancestors[1:]:
                meta = node.data(Qt.ItemDataRole.UserRole + 2)
                if meta is not None:
                    # Validate metadata structure and key index
                    if (isinstance(meta, tuple) and len(meta) == 2):
                        parent_dict, key_index = meta
                        if isinstance(parent_dict, dict):
                            keys_list = list(parent_dict.keys())
                            if 0 <= key_index < len(keys_list):
                                key_obj = keys_list[key_index]
                            else:
                                key_obj = "<invalid_key_index>"
                        else:
                            key_obj = "<invalid_parent_dict>"

                        tmp_var_name = f"tmp_key_{tmp_var_count}"
                        tmp_var_count += 1
                        lines.append(f"{tmp_var_name} = {repr(key_obj)}")
                        current_expr += f"[{tmp_var_name}]"
                    else:
                        current_expr += f".{node.text()}"
                else:
                    parent = node.parent()
                    parent_data = parent.data(
                        Qt.ItemDataRole.UserRole + 1) if parent else None
                    if parent is not None and isinstance(parent_data, dict):
                        key_text = node.text()
                        current_expr += f"[{repr(key_text)}]"
                    else:
                        current_expr += f".{node.text()}"

            if not (current_expr.startswith(
                    f"{self.alias}.") or current_expr.startswith(f"{self.alias}[")):
                current_expr = f"{self.alias}.{current_expr}"

            lines.append(current_expr)
            return lines

        except Exception as e:
            logger.error(
                f"Error computing multiline path for item '{item.text()}': {e}")
            # Fallback to simpler resolution on error
            path = self.viewer.resolve_item_path(item)
            if path and not (path.startswith(f"{self.alias}.") or path.startswith(
                    f"{self.alias}[")):
                path = f"{self.alias}.{path}"
            return [path]

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
