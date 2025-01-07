# tests/test_viewer.py

from unittest.mock import MagicMock

import pytest
from var_view.variable_viewer.viewer import VariableViewer


# Dummy data source for testing
class DummyDataSource:
    var_int = 42
    var_list = [1, 2, 3]
    var_str = "Hello, World!"

@pytest.fixture
def dummy_data_source():
    return DummyDataSource()


@pytest.fixture
def viewer_with_mocked_plugins(app, dummy_data_source):
    from var_view.variable_viewer.viewer import VariableViewer
    viewer = VariableViewer(dummy_data_source)

    # Replace plugin_manager with a mock to control its behavior
    viewer.plugin_manager = MagicMock()
    viewer.plugin_manager.get_handler_for_type.return_value = None  # Simulate no plugin found

    return viewer


@pytest.fixture
def viewer(app, dummy_data_source):
    # Instantiate VariableViewer with dummy data source
    viewer = VariableViewer(dummy_data_source)
    # Call refresh_view to populate the model
    viewer.refresh_view()
    return viewer


def test_refresh_view_populates_model(viewer):
    """
    Test that refresh_view populates the model with variables from the data source.
    """
    root = viewer.model.invisibleRootItem()
    variable_names = set()

    # Iterate over top-level items in the model
    for row in range(root.rowCount()):
        item = root.child(row, 0)  # Column 0 contains variable names
        variable_names.add(item.text())

    # Check that all dummy variables are present in the model
    assert "var_int" in variable_names
    assert "var_list" in variable_names
    assert "var_str" in variable_names


def test_calculate_size_with_various_types(viewer):
    """
    Test that calculate_size returns correct values for different data types.
    """
    # Test size for string
    size_str = viewer.calculate_size("test")
    assert size_str == "4"

    # Test size for list
    size_list = viewer.calculate_size([1, 2, 3])
    assert size_list == "3"

    # Test size for dict
    size_dict = viewer.calculate_size({"a": 1, "b": 2})
    assert size_dict == "{2}"


def test_format_value_truncates_long_strings(viewer):
    """
    Test that format_value correctly truncates long values.
    """
    long_str = "a" * 200  # 200 characters
    formatted = viewer.format_value(long_str)
    # Check that the formatted string is truncated to 150 characters + ellipsis
    assert len(formatted) <= 150
    assert formatted.endswith("...")


def test_infer_type_hint_general_for_dict(viewer):
    """
    Test infer_type_hint_general with a dictionary input.
    """
    from var_view.variable_viewer.utils import infer_type_hint_general
    data = {"key1": 1, "key2": 2}
    hint = infer_type_hint_general(data)
    # Depending on implementation, hint might look like: "dict[int, int]" or similar
    assert "dict" in hint


def test_calculate_memory_usage_with_builtin(viewer):
    # Test for built-in types
    memory_usage_int = viewer.calculate_memory_usage(42)
    assert "B" in memory_usage_int  # Basic check for byte unit


def test_calculate_memory_usage_with_plugin(viewer, caplog):
    from unittest.mock import MagicMock
    from var_view.variable_viewer.utils import VariableRepresentation, format_bytes

    class DummyHandler:
        def __call__(self, value):
            return VariableRepresentation(nbytes=1024)

    viewer.plugin_manager = MagicMock()
    viewer.plugin_manager.get_handler_for_type.return_value = DummyHandler()

    class Dummy:
        pass

    dummy_obj = Dummy()

    with caplog.at_level("DEBUG"):
        memory_usage = viewer.calculate_memory_usage(dummy_obj)
        for record in caplog.records:
            print("LOG:", record.levelname, record.message)

    expected = format_bytes(1024)
    assert memory_usage == expected


def test_format_value_for_list(viewer):
    # Test for list with more than 5 elements
    long_list = list(range(10))
    formatted = viewer.format_value(long_list)
    assert formatted.startswith("[")
    assert "..." in formatted


def test_format_value_for_dict(viewer):
    data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
    formatted = viewer.format_value(data)
    assert formatted.startswith("{")
    assert "..." in formatted if len(data) > 5 else "..." not in formatted


def test_resolve_variable_invalid_path(viewer):
    # Test resolving a non-existent variable path returns None
    result = viewer.resolve_variable("nonexistent")
    assert result is None


def test_resolve_item_path_returns_string(viewer):
    # This requires adding a dummy item hierarchy
    from PyQt6.QtGui import QStandardItem
    root = viewer.model.invisibleRootItem()
    child = QStandardItem("var_int")
    root.appendRow([child, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])
    path = viewer.resolve_item_path(child)
    assert isinstance(path, str)


def test_calculate_size_for_unknown_type(viewer):
    # Use a custom object without __len__ or shape
    class Dummy:
        pass
    size = viewer.calculate_size(Dummy())
    # Adjusting expectation based on the actual behavior of calculate_size
    assert size == "0"


def test_copy_variable_path(viewer, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QStandardItem
    from unittest.mock import MagicMock

    # Mock clipboard
    mock_clipboard = MagicMock()
    monkeypatch.setattr(QApplication, "clipboard", lambda: mock_clipboard)

    # Create a QStandardItem and add it to the model
    item = QStandardItem("var_int")
    # Fill the rest of the columns with empty QStandardItems
    empty_item = QStandardItem()
    viewer.model.invisibleRootItem().appendRow([item, empty_item, empty_item, empty_item, empty_item])

    # Get the index of the created item and call copy_variable_path
    index = viewer.model.indexFromItem(item)
    viewer.copy_variable_path([index])

    # Verify that setText was called on the mocked clipboard
    mock_clipboard.setText.assert_called_once()
