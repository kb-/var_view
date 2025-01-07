import pytest
from unittest.mock import MagicMock
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem

from var_view import VariableViewer
from var_view.variable_viewer.model import VariableStandardItemModel


@pytest.fixture
def viewer():
    """
    Mocked viewer fixture for tests.
    """
    mock_viewer = MagicMock(spec=VariableViewer)
    mock_viewer.resolve_item_path = MagicMock(return_value="resolved_path")
    return mock_viewer


@pytest.fixture
def model_with_viewer(viewer):
    """
    Fixture for creating a VariableStandardItemModel with a mocked viewer.
    """
    viewer.resolve_item_path = MagicMock(return_value="resolved_path")
    return VariableStandardItemModel(viewer)


def test_supportedDragActions(model_with_viewer):
    """
    Test that supportedDragActions returns Qt.DropAction.CopyAction.
    """
    assert model_with_viewer.supportedDragActions() == Qt.DropAction.CopyAction


def test_mimeData_includes_correct_paths(model_with_viewer):
    """
    Test that mimeData includes correct paths for variables in column 0.
    """
    # Setup model with a single variable
    root = model_with_viewer.invisibleRootItem()
    item = QStandardItem("var_name")
    root.appendRow([item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    # Simulate mimeData call
    index = model_with_viewer.indexFromItem(item)
    mime_data = model_with_viewer.mimeData([index])

    # Verify paths are included and prefixed with alias
    expected_path = f"{model_with_viewer.alias}.resolved_path"
    assert mime_data.text() == expected_path
    model_with_viewer.viewer.resolve_item_path.assert_called_once_with(item)


def test_mimeData_skips_non_column_zero(model_with_viewer):
    """
    Test that mimeData skips variables not in column 0.
    """
    # Setup model with a variable
    root = model_with_viewer.invisibleRootItem()
    item = QStandardItem("var_name")
    root.appendRow([item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    # Create index for column 1 (non-zero column)
    index = model_with_viewer.index(0, 1)  # Row 0, Column 1
    mime_data = model_with_viewer.mimeData([index])

    # Verify that no paths are included
    assert mime_data.text() == ""


def test_mimeData_without_alias_prefix(model_with_viewer):
    """
    Test that mimeData includes paths without alias prefix if already prefixed.
    """
    # Mock resolve_item_path to return a prefixed path
    model_with_viewer.viewer.resolve_item_path = MagicMock(return_value=f"{model_with_viewer.alias}.prefixed_path")

    # Setup model with a single variable
    root = model_with_viewer.invisibleRootItem()
    item = QStandardItem("var_name")
    root.appendRow([item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    # Simulate mimeData call
    index = model_with_viewer.indexFromItem(item)
    mime_data = model_with_viewer.mimeData([index])

    # Verify that the path is not double-prefixed
    expected_path = f"{model_with_viewer.alias}.prefixed_path"
    assert mime_data.text() == expected_path


def test_flags_for_valid_index(model_with_viewer):
    """
    Test that flags for a valid index include drag enabled for column 0.
    """
    # Setup model with a single variable
    root = model_with_viewer.invisibleRootItem()
    item = QStandardItem("var_name")
    root.appendRow([item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    # Create index for column 0
    index = model_with_viewer.indexFromItem(item)

    # Verify that drag is enabled
    assert model_with_viewer.flags(index) & Qt.ItemFlag.ItemIsDragEnabled


# def test_flags_for_non_column_zero(model_with_viewer):
#     """
#     Test that flags for non-column 0 do not include drag enabled.
#     """
#     # Setup model with a single variable
#     root = model_with_viewer.invisibleRootItem()
#     item = QStandardItem("var_name")
#     root.appendRow([item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])
#
#     # Create index for column 1 (non-zero column)
#     index = model_with_viewer.index(0, 1)  # Row 0, Column 1
#
#     # Verify that drag is not enabled
#     assert not (model_with_viewer.flags(index) & Qt.ItemFlag.ItemIsDragEnabled)


def test_flags_for_invalid_index(model_with_viewer):
    """
    Test that flags for an invalid index return Qt.NoItemFlags.
    """
    invalid_index = model_with_viewer.index(-1, -1)  # Invalid index
    assert model_with_viewer.flags(invalid_index) == Qt.ItemFlag.NoItemFlags
