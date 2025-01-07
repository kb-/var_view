import pytest
from PyQt6.QtCore import Qt, QModelIndex
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from var_view.variable_viewer.model import VariableStandardItemModel

# Create a simple dummy viewer to pass to the model.
class DummyViewer:
    alias = "data_source"
    def resolve_item_path(self, item):
        # For testing, always return a fixed path.
        return "resolved_path"

@pytest.fixture
def model_with_viewer():
    # Initialize the model with a dummy viewer.
    viewer = DummyViewer()
    model = VariableStandardItemModel(viewer, alias=viewer.alias)
    # Assign viewer to model explicitly for convenience.
    model.viewer = viewer
    return model


def test_supportedDragActions(model_with_viewer):
    assert model_with_viewer.supportedDragActions() == Qt.DropAction.CopyAction


def test_flags_for_valid_index(model_with_viewer):
    # Create a row with 5 columns.
    root = model_with_viewer.invisibleRootItem()
    items = [QStandardItem(f"col{i}") for i in range(5)]
    root.appendRow(items)

    # Test that all columns have the ItemIsDragEnabled flag.
    for column in range(5):
        index = model_with_viewer.index(0, column)
        flags = model_with_viewer.flags(index)
        assert flags & Qt.ItemFlag.ItemIsDragEnabled


def test_flags_for_invalid_index(model_with_viewer):
    # For an invalid index, flags should return NoItemFlags.
    invalid_index = QModelIndex()
    flags = model_with_viewer.flags(invalid_index)
    assert flags == Qt.ItemFlag.NoItemFlags


def test_mimeData_includes_correct_paths(model_with_viewer):
    root = model_with_viewer.invisibleRootItem()
    # Create an item in column 0.
    item = QStandardItem("var_name")
    # Append a row with five columns.
    row = [item] + [QStandardItem() for _ in range(4)]
    root.appendRow(row)

    index = model_with_viewer.indexFromItem(item)
    mime_data = model_with_viewer.mimeData([index])
    expected = f"{model_with_viewer.viewer.alias}.resolved_path"
    assert mime_data.text() == expected

def test_mimeData_skips_non_column_zero(model_with_viewer):
    root = model_with_viewer.invisibleRootItem()
    # Create an item in column 1.
    items = [QStandardItem("var_name") for _ in range(5)]
    root.appendRow(items)

    # Create an index for column 1.
    index = model_with_viewer.index(0, 1)
    mime_data = model_with_viewer.mimeData([index])
    # Because mimeData only processes column 0, we expect an empty text.
    assert mime_data.text() == ""

def test_mimeData_without_alias_prefix(model_with_viewer):
    # Override resolve_item_path to return a path that already starts with alias.
    original_resolve = model_with_viewer.viewer.resolve_item_path
    model_with_viewer.viewer.resolve_item_path = lambda item: f"{model_with_viewer.viewer.alias}.already_prefixed"

    root = model_with_viewer.invisibleRootItem()
    item = QStandardItem("var_name")
    row = [item] + [QStandardItem() for _ in range(4)]
    root.appendRow(row)

    index = model_with_viewer.indexFromItem(item)
    mime_data = model_with_viewer.mimeData([index])
    expected = f"{model_with_viewer.viewer.alias}.already_prefixed"
    assert mime_data.text() == expected

    model_with_viewer.viewer.resolve_item_path = original_resolve  # Restore original method
