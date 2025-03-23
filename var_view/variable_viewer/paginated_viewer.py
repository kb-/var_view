from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItem
from var_view.variable_viewer.viewer import VariableViewer, ObjectRef
import logging

def item_path(item):
    """Reconstruct a dotted path up the tree, e.g. '[99].[3].nested_dict'."""
    parts = []
    cur = item
    while cur is not None:
        parts.append(cur.text())
        cur = cur.parent()
    parts.reverse()
    return ".".join(parts)

class PaginatedVariableViewer(VariableViewer):
    def __init__(self, data_source, alias="data_source", app_plugin_dir=None):
        self.batch_size = 100  # 100 items per container
        super().__init__(data_source, alias, app_plugin_dir)
        self.tree_view.doubleClicked.connect(self.handle_double_click)

    def load_children(self, parent_item, value, visited=None, start_index=0):
        """
        Overrides load_children to load children in batches.
        If the container (list or dict) has more items than batch_size,
        only a batch is loaded and a 'Load More' item is appended.
        """
        if visited is None:
            visited = set()

        # Remove cycle detection: do not check for id(value) in visited.
        # (Optionally, you might add a recursion depth check if needed.)
        # visited.add(id(value))  <-- REMOVED

        # For list-type containers:
        if isinstance(value, list):
            total = len(value)
            end_index = min(start_index + self.batch_size, total)

            for i in range(start_index, end_index):
                elem = value[i]
                # Use your existing add_variable to create the row.
                self.add_variable(f"[{i}]", elem, parent_item)

            if end_index < total:
                load_more_item = QStandardItem(f"... Load More ({end_index}/{total})")
                load_more_item.setEditable(False)
                load_more_item.setForeground(Qt.GlobalColor.blue)
                load_more_item.setData({
                    'is_load_more': True,
                    'container_type': 'list',
                    'value_ref': value,
                    'offset': end_index,
                    'visited': visited  # still pass visited if needed downstream
                }, Qt.ItemDataRole.UserRole)
                parent_item.appendRow([
                    load_more_item,
                    QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()
                ])

        # For dict-type containers:
        elif isinstance(value, dict):
            # Store a stable key list for consistent ordering.
            stored_data = parent_item.data(Qt.ItemDataRole.UserRole + 10)
            if stored_data and "dict_keys" in stored_data:
                dict_keys = stored_data["dict_keys"]
            else:
                dict_keys = list(value.keys())
                parent_item.setData({"dict_keys": dict_keys}, Qt.ItemDataRole.UserRole + 10)

            total = len(dict_keys)
            end_index = min(start_index + self.batch_size, total)

            for i in range(start_index, end_index):
                key = dict_keys[i]
                val = value[key]
                self.add_variable(str(key), val, parent_item)

            if end_index < total:
                load_more_item = QStandardItem(f"... Load More ({end_index}/{total})")
                load_more_item.setEditable(False)
                load_more_item.setForeground(Qt.GlobalColor.blue)
                load_more_item.setData({
                    'is_load_more': True,
                    'container_type': 'dict',
                    'value_ref': value,
                    'offset': end_index,
                    'visited': visited
                }, Qt.ItemDataRole.UserRole)
                parent_item.appendRow([
                    load_more_item,
                    QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()
                ])
            return

        else:
            # For non-list/dict types, use the existing logic.
            super().load_children(parent_item, value, visited)

    def handle_double_click(self, index):
        """
        Detects double-click on a 'Load More' item and loads the next batch.
        """
        item = self.model.itemFromIndex(index)
        if not item:
            return

        user_data = item.data(Qt.ItemDataRole.UserRole)
        if user_data and isinstance(user_data, dict) and user_data.get('is_load_more'):
            parent = item.parent()
            if parent is None:
                parent = self.model.invisibleRootItem()
                self.model.removeRow(item.row())
            else:
                parent.removeRow(item.row())

            container_type = user_data['container_type']
            value_ref = user_data['value_ref']
            offset = user_data['offset']
            visited = user_data['visited']
            self.load_children(parent, value_ref, visited, start_index=offset)
        else:
            # Otherwise, do nothing special.
            pass
