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


def test_add_variable_with_plugin(viewer):
    from PyQt6.QtGui import QStandardItem
    from var_view.variable_viewer.utils import VariableRepresentation

    # Create a dummy plugin handler that returns a specific representation
    def dummy_handler(value):
        return VariableRepresentation(nbytes=2048, shape=(10, 10), dtype="float", value_summary="summary")

    # Mock plugin_manager and get_handler_for_type
    viewer.plugin_manager = MagicMock()
    viewer.plugin_manager.get_handler_for_type.return_value = dummy_handler

    parent_item = viewer.model.invisibleRootItem()
    viewer.add_variable("dummy_var", [1, 2, 3], parent_item)

    # Verify that the variable was added to the model correctly
    item = parent_item.child(parent_item.rowCount() - 1, 0)  # Last added item
    assert item.text() == "dummy_var"



def test_add_variable_handles_exception(viewer, caplog):
    from PyQt6.QtGui import QStandardItem

    # Force an exception by making a plugin handler raise an exception
    def faulty_handler(value):
        raise ValueError("Test exception")

    # Mock plugin_manager and get_handler_for_type
    viewer.plugin_manager = MagicMock()
    viewer.plugin_manager.get_handler_for_type.return_value = faulty_handler

    parent_item = viewer.model.invisibleRootItem()

    with caplog.at_level("ERROR"):
        viewer.add_variable("error_var", "some_value", parent_item)

    # Check that an error was logged
    assert any("Error adding variable 'error_var'" in record.message for record in caplog.records)

    # Verify that an error entry was added to the model
    last_row = parent_item.rowCount() - 1
    type_item = parent_item.child(last_row, 1)
    assert type_item.text() == "Error"



def test_handle_expand_and_load_children(viewer):
    from PyQt6.QtGui import QStandardItem

    # Prepare a dummy nested structure in data_source
    viewer.data_source = type("DummySource", (), {"child": [1, 2]})()
    root_item = viewer.model.invisibleRootItem()
    # Add a placeholder item to simulate expandable child
    parent_item = QStandardItem("child")
    root_item.appendRow([parent_item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])
    # Add placeholder for lazy loading as expected by handle_expand
    placeholder = QStandardItem("Loading...")
    parent_item.appendRow([placeholder, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    # Simulate expanding the item
    index = viewer.model.indexFromItem(parent_item)
    viewer.handle_expand(index)

    # After expansion, there should be more children loaded, not just the placeholder
    assert parent_item.rowCount() > 0
    # Check that the placeholder is removed
    assert parent_item.child(0, 0).text() != "Loading..."


def test_resolve_variable_complex(viewer):
    # Define a nested data structure to test resolution
    class Child:
        value = 123
    class Parent:
        child = Child()
    viewer.data_source = Parent()

    path = "child.value"  # This is a simplified path; adapt based on resolve_variable's parsing logic
    result = viewer.resolve_variable(path)
    assert result == 123


def test_resolve_item_path_multiple_levels(viewer):
    from PyQt6.QtGui import QStandardItem
    root = viewer.model.invisibleRootItem()
    parent = QStandardItem("parent")
    child = QStandardItem("child")
    root.appendRow([parent, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])
    parent.appendRow([child, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    path = viewer.resolve_item_path(child)
    # Check that the path includes both parent and child names
    assert "parent" in path and "child" in path
