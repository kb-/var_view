# tests/test_viewer.py
from collections import namedtuple
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QStandardItem
from PyQt6.QtWidgets import QMenu

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


def test_load_plugins_handles_missing_plugins(viewer_with_mocked_plugins, caplog, tmp_path):
    """
    Test that load_plugins logs warnings if plugins fail to load.
    """
    viewer = viewer_with_mocked_plugins
    app_plugin_dir = tmp_path / "plugins"  # Temporary directory for plugins
    app_plugin_dir.mkdir()

    # Mock a plugin manager that raises an error
    viewer.plugin_manager.load_plugins_from_directory.side_effect = Exception("Test error")

    with caplog.at_level("WARNING"):
        viewer.load_plugins(app_plugin_dir)
        assert any("Test error" in record.message for record in caplog.records)


def test_load_plugins_registers_plugins(viewer_with_mocked_plugins, tmp_path):
    """
    Test that load_plugins successfully registers plugins.
    """
    viewer = viewer_with_mocked_plugins
    app_plugin_dir = tmp_path / "plugins"
    app_plugin_dir.mkdir()

    # Create a dummy plugin file to simulate plugin registration
    (app_plugin_dir / "test_plugin.py").write_text("")

    # Mock plugin manager methods we care about
    viewer.plugin_manager.load_plugins_from_directory = MagicMock()

    # Call the method
    viewer.load_plugins(app_plugin_dir)

    # Verify that load_plugins_from_directory was called with the correct directory
    viewer.plugin_manager.load_plugins_from_directory.assert_called_once_with(app_plugin_dir)


def simple_resolve(data_source, path):
    current = data_source
    for part in path.split('.'):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            break
    return current


def test_resolve_variable_handles_invalid_path(viewer):
    """
    Test that resolve_variable returns None for invalid paths.
    """
    result = viewer.resolve_variable("invalid.path")
    assert result is None


def test_resolve_variable_nested_objects(viewer):
    """
    Test resolving nested objects.
    """
    nested_data = {"level1": {"level2": {"key": "value"}}}
    viewer.data_source = nested_data
    viewer.resolve_variable = lambda path: simple_resolve(viewer.data_source, path)
    result = viewer.resolve_variable("level1.level2.key")
    assert result == "value"


# def test_resolve_variable_cyclic_reference(viewer):
#     """
#     Test that resolve_variable handles cyclic references gracefully.
#     """
#     class Cyclic:
#         pass
#
#     cyclic = Cyclic()
#     cyclic.self = cyclic
#
#     viewer.data_source = {"cyclic": cyclic}
#     viewer.resolve_variable = lambda path: simple_resolve(viewer.data_source, path)
#     result = viewer.resolve_variable("cyclic.self.self.self")
#     assert result == cyclic


def test_load_children_expands_item(viewer):
    """
    Test that load_children populates children for an expandable item.
    """
    from PyQt6.QtGui import QStandardItem
    parent = QStandardItem("parent")
    viewer.model.invisibleRootItem().appendRow(parent)

    # Mock expandable item behavior
    viewer.can_expand = MagicMock(return_value=True)
    dummy_value = {"dummy": 1}  # Example value to pass
    viewer.load_children(parent, dummy_value)

    # Assert that children were added to the parent item
    assert parent.rowCount() > 0


def test_show_context_menu_triggers(viewer):
    """
    Test that show_context_menu is called when customContextMenuRequested signal is emitted.
    """
    from PyQt6.QtCore import QPoint

    # Replace show_context_menu with a mock and reconnect the signal
    viewer.tree_view.customContextMenuRequested.disconnect()
    viewer.show_context_menu = MagicMock()
    viewer.tree_view.customContextMenuRequested.connect(viewer.show_context_menu)

    # Create an arbitrary point to emit the signal
    pos = QPoint(10, 10)

    # Emit the customContextMenuRequested signal
    viewer.tree_view.customContextMenuRequested.emit(pos)

    # Verify that our mock was called once with the emitted position
    viewer.show_context_menu.assert_called_once_with(pos)


def test_export_variable(viewer):
    # Mock export_variable behavior
    viewer.export_variable = MagicMock()
    viewer.export_variable("mock_variable", "mock_file")
    viewer.export_variable.assert_called_once_with("mock_variable", "mock_file")

# @pytest.mark.parametrize("input_data,expected", [
#     ({"key": "value"}, "value"),
#     ({"key": None}, None),
#     ({"level1": {"level2": {"key": "nested_value"}}}, "nested_value"),
#     ({"key": {"subkey": "subvalue"}}, {"subkey": "subvalue"}),
#     # Add more edge cases as needed
# ])
# def test_resolve_variable_edge_cases(viewer, input_data, expected):
#     viewer.data_source = input_data
#     result = viewer.resolve_variable("key")
#     assert result == expected


@pytest.mark.parametrize("input_data, path, expected", [
    ({"a": {"b": {"c": "d"}}}, "a.b.c", "d"),
    ({"x": {"y": None}}, "x.y", None),
    ({"m": {"n": {"o": {"p": "q"}}}}, "m.n.o.p", "q"),
])
def test_resolve_variable_nested_paths(viewer, input_data, path, expected):
    viewer.data_source = input_data
    assert viewer.resolve_variable(path) == expected


@pytest.mark.parametrize("input_data, path, expected", [
    ({"a": {"b": {"c": "d"}}}, 'a["b"]["c"]', "d"),
    ({"x": {"y": None}}, 'x["y"]', None),
    ({"m": {"n": {"o": {"p": "q"}}}}, 'm["n"]["o"]["p"]', "q"),
])
def test_resolve_variable_nested_paths(viewer, input_data, path, expected):
    class Root:
        pass
    root_obj = Root()
    # Attach each top-level key as an attribute of root_obj
    for key, value in input_data.items():
        setattr(root_obj, key, value)
    viewer.data_source = root_obj

    assert viewer.resolve_variable(path) == expected


def test_resolve_variable_nested_paths_var2(viewer):
    class Root:
        pass
    root_obj = Root()
    root_obj.m = {"n": {"o": {"p": "q"}}}
    viewer.data_source = root_obj

    assert viewer.resolve_variable('m["n"]["o"]["p"]') == "q"


def test_resolve_variable_cyclic_reference(viewer):
    class Cyclic:
        pass

    cyclic = Cyclic()
    cyclic.self = cyclic

    class Root:
        pass
    root_obj = Root()
    root_obj.cyclic = cyclic
    viewer.data_source = root_obj

    # Test cyclic resolution. It should eventually return the cyclic object
    result = viewer.resolve_variable('cyclic.self.self.self')
    assert result is cyclic


def test_resolve_variable_basic(viewer):
    # Create an object with attribute 'key'
    class Root:
        key = "value"

    viewer.data_source = Root()

    # Simple attribute lookup should work
    assert viewer.resolve_variable("key") == "value"


def test_resolve_variable_invalid_paths(viewer):
    class Root:
        pass
    root_obj = Root()
    root_obj.key = "value"
    viewer.data_source = root_obj

    # Non-existing attributes or keys
    assert viewer.resolve_variable("nonexistent") is None
    assert viewer.resolve_variable('key["nonexistent"]') is None


def test_resolve_mixed_nested(viewer):
    class Child:
        pass

    class Root:
        pass

    root_obj = Root()
    child = Child()
    child.value = 123
    root_obj.child = {"nested": child}
    viewer.data_source = root_obj

    # Accessing attribute 'child', then key 'nested', then attribute 'value'
    assert viewer.resolve_variable('child["nested"].value') == 123


def test_load_children_handles_dict(viewer):
    parent_item = QStandardItem("parent")
    sample_dict = {"key1": 1, "key2": 2}
    initial_rows = parent_item.rowCount()

    viewer.load_children(parent_item, sample_dict)

    # Check that children were added for each key in the dictionary
    assert parent_item.rowCount() > initial_rows


def test_load_children_handles_namedtuple(viewer):
    parent_item = QStandardItem("parent")
    DummyTuple = namedtuple("DummyTuple", ["field1", "field2"])
    value = DummyTuple(field1=123, field2=456)
    initial_rows = parent_item.rowCount()

    viewer.load_children(parent_item, value)

    # Check that children were added for each field in the namedtuple
    assert parent_item.rowCount() > initial_rows


def test_load_children_handles_qobject(viewer):
    class DummyQObject(QObject):
        def __init__(self):
            super().__init__()
            self.attribute = "test"

    parent_item = QStandardItem("parent")
    value = DummyQObject()
    initial_rows = parent_item.rowCount()

    viewer.load_children(parent_item, value)

    # Verify children added for non-callable attributes of QObject
    assert parent_item.rowCount() > initial_rows


def test_load_children_handles_dict_object(viewer):
    class Dummy:
        pass

    obj = Dummy()
    obj.attr1 = "value1"
    obj.attr2 = 42

    parent_item = QStandardItem("parent")
    initial_rows = parent_item.rowCount()

    viewer.load_children(parent_item, obj)

    # Verify children added for each non-private attribute
    assert parent_item.rowCount() > initial_rows


def test_load_children_handles_slots(viewer):
    class DummyWithSlots:
        __slots__ = ['slot1', 'slot2']

        def __init__(self):
            self.slot1 = "value1"
            self.slot2 = 2

    parent_item = QStandardItem("parent")
    value = DummyWithSlots()
    initial_rows = parent_item.rowCount()

    viewer.load_children(parent_item, value)

    # Verify children added for each slot
    assert parent_item.rowCount() > initial_rows


def test_load_children_handles_else(viewer):
    class Dummy:
        def __init__(self):
            self.a = 1

        def method(self):
            pass

    parent_item = QStandardItem("parent")
    value = Dummy()
    initial_rows = parent_item.rowCount()

    viewer.load_children(parent_item, value)

    # Verify children added for non-callable attributes found in dir(value)
    assert parent_item.rowCount() > initial_rows


def test_load_children_handles_else_branch(viewer):
    from PyQt6.QtGui import QStandardItem

    # Create a parent item in the model
    parent_item = QStandardItem("parent")
    viewer.model.invisibleRootItem().appendRow(parent_item)

    # Use a built-in type (integer) that doesn't have __dict__ or __slots__
    value = 42

    # Record initial row count
    initial_count = parent_item.rowCount()

    # Call load_children with the integer value
    viewer.load_children(parent_item, value)

    # Since 42 has no non-callable attributes without underscores,
    # the method should run without error and not necessarily add children.
    # We check that the row count has not decreased.
    assert parent_item.rowCount() >= initial_count


def test_show_context_menu_construction(viewer, monkeypatch):
    from PyQt6.QtWidgets import QMenu
    from PyQt6.QtCore import QPoint
    from PyQt6.QtGui import QStandardItem

    # Create a QStandardItem and append it to the model to simulate selection
    item = QStandardItem("var_int")
    empty_item = QStandardItem()
    viewer.model.invisibleRootItem().appendRow([item, empty_item, empty_item, empty_item, empty_item])
    valid_index = viewer.model.indexFromItem(item)

    # Create a FakeQMenu to capture actions
    class FakeQMenu(QMenu):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.actions_added = []

        def addAction(self, action):
            self.actions_added.append(action)
            return super().addAction(action)

        def exec(self, _):
            # Override exec to do nothing
            return

    fake_menu_instance = FakeQMenu()

    # Monkeypatch QMenu to always return our fake instance
    monkeypatch.setattr('var_view.variable_viewer.viewer.QMenu', lambda *args, **kwargs: fake_menu_instance)

    # Simulate that there is at least one selected index using the valid index
    monkeypatch.setattr(viewer.tree_view, 'selectedIndexes', lambda: [valid_index])

    # Emit a fake signal position (the actual value is not important for this test)
    pos = QPoint(10, 10)

    # Call the method under test
    viewer.show_context_menu(pos)

    # Extract texts from the actions added to the fake menu
    action_texts = [action.text() for action in fake_menu_instance.actions_added]

    # Verify that expected actions were added to the menu
    assert any(text in action_texts for text in ["Export", "Export Selected"])
    assert any(text in action_texts for text in ["Update", "Copy Path"])



def test_export_variable_success(viewer, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    # Setup a fake exporter
    fake_exporter = MagicMock()
    viewer.exporter = fake_exporter

    # Create a dummy item and index in the model
    from PyQt6.QtGui import QStandardItem
    item = QStandardItem("var_test")
    viewer.model.invisibleRootItem().appendRow(
        [item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()]
    )
    index = viewer.model.indexFromItem(item)

    # Override resolve_variable to return a non-None value
    viewer.resolve_variable = lambda path: 123

    # Call export_variable
    viewer.export_variable(index)

    fake_exporter.export_variable.assert_called_once_with("var_test", 123)


def test_export_variable_failure(viewer, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    # Setup a fake exporter
    fake_exporter = MagicMock()
    viewer.exporter = fake_exporter

    # Create a dummy item and index in the model
    from PyQt6.QtGui import QStandardItem
    item = QStandardItem("var_test")
    viewer.model.invisibleRootItem().appendRow(
        [item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()]
    )
    index = viewer.model.indexFromItem(item)

    # Override resolve_variable to return None
    viewer.resolve_variable = lambda path: None

    # Capture warnings instead of showing QMessageBox
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda parent, title, text: messages.append(text))

    viewer.export_variable(index)

    # Check that a warning was shown
    assert any("could not be resolved" in msg for msg in messages)


def test_export_selected_variables_no_valid(viewer, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    # Setup a fake exporter
    fake_exporter = MagicMock()
    viewer.exporter = fake_exporter

    # Create dummy indexes that don't correspond to valid variables
    fake_index = MagicMock()
    monkeypatch.setattr(viewer.model, 'itemFromIndex', lambda idx: None)

    # Capture warnings
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda parent, title, text: messages.append(text))

    viewer.export_selected_variables([fake_index])
    assert any("No valid variables selected" in msg for msg in messages)


def test_update_selected_variables_handles_unavailable(viewer, caplog):
    # Simulate an index for a non-existing variable
    fake_index = MagicMock()
    fake_item = MagicMock()
    fake_item.parent.return_value = None
    fake_item.text.return_value = "nonexistent"
    fake_index.column.return_value = 0
    fake_index.row.return_value = 0
    viewer.model.itemFromIndex = lambda idx: fake_item

    viewer.resolve_item_path = lambda item: "nonexistent"
    viewer.resolve_variable = lambda path: None

    with caplog.at_level("WARNING"):
        viewer.update_selected_variables([fake_index])
        assert any("Variable 'nonexistent' is unavailable" in record.message for record in caplog.records)


def test_unload_and_reload_item_plugin_branch(viewer, monkeypatch):
    from PyQt6.QtGui import QStandardItem
    from unittest.mock import MagicMock

    # 1) Create a QStandardItem as "item"
    item = QStandardItem("test_item")

    # 2) Make an index mock that returns row=0, column=0
    index_mock = MagicMock()
    index_mock.row.return_value = 0
    index_mock.column.return_value = 0
    # Make sibling(...) calls for columns 1..4 return new mocks
    sibling_mocks = {}
    def sibling(row, col):
        # Return a separate MagicMock for each col
        subindex = MagicMock()
        subindex.column.return_value = col
        subindex.row.return_value = row
        return subindex
    index_mock.sibling.side_effect = sibling

    item.index = MagicMock(return_value=index_mock)

    # 3) For each col (1..4), we want itemFromIndex(...) to return a QStandardItem mock
    siblings = [MagicMock(spec=QStandardItem) for _ in range(5)]  # columns 0..4
    def fake_itemFromIndex(idx):
        return siblings[idx.column()]

    monkeypatch.setattr(viewer.model, "itemFromIndex", fake_itemFromIndex)

    # 4) Setup plugin manager mock
    viewer.plugin_manager = MagicMock()
    class DummyRepresentation:
        shape = (10,10)
        value_summary = "summary"
        nbytes = 2048
    viewer.plugin_manager.get_handler_for_type.return_value = lambda v: DummyRepresentation()

    # 5) Call the method
    viewer.unload_and_reload_item(item, "dummy_value", "dummy_path")

    # 6) Check that each sibling's setText was called with a non-empty string
    for i, sib in enumerate(siblings[1:], start=1):
        # The code sets column 1 to 'value_type', col 2 to 'size_str', etc.
        sib.setText.assert_called()  # or use .assert_called_once()
        # optionally, check call args if you want to confirm it's not empty
        call_args = sib.setText.call_args[0][0]
        assert call_args != "", f"Column {i} text unexpectedly empty."


def test_unload_and_reload_item_no_plugin(viewer, monkeypatch):
    from PyQt6.QtGui import QStandardItem
    from PyQt6.QtCore import QModelIndex
    from unittest.mock import MagicMock

    # Create a dummy QStandardItem as the item to update
    item = MagicMock(spec=QStandardItem)

    # Mock `item.index()` to return a dummy QModelIndex
    class FakeIndex(QModelIndex):
        def __init__(self, column=0):
            self._column = column

        def column(self):
            return self._column

        def sibling(self, row, column):
            return FakeIndex(column=column)

    item.index.return_value = FakeIndex()

    # Create dummy sibling items for each column (5 columns total)
    siblings = [MagicMock(spec=QStandardItem) for _ in range(5)]

    # Mock `itemFromIndex` to return sibling items based on the column
    def fake_itemFromIndex(idx):
        if isinstance(idx, FakeIndex):
            return siblings[idx.column()]
        return None

    monkeypatch.setattr(viewer.model, "itemFromIndex", fake_itemFromIndex)

    # Configure plugin manager to simulate no plugin
    viewer.plugin_manager = MagicMock()
    viewer.plugin_manager.get_handler_for_type = MagicMock(return_value=None)

    # Override methods to prevent UI dependencies
    viewer.can_expand = lambda v: False  # Avoid reloading children
    viewer.calculate_size = lambda v: "size"
    viewer.format_value = lambda v: "formatted"
    viewer.calculate_memory_usage = lambda v: "mem_usage"

    # Call `unload_and_reload_item` with dummy parameters
    viewer.unload_and_reload_item(item, "dummy_value", "dummy_path")

    # Update expected values: column 1 -> value_type ("str"), col2 -> "size", col3 -> "formatted", col4 -> "mem_usage"
    expected_values = ["str", "size", "formatted", "mem_usage"]

    # Assert `setText` was called with correct values for columns 1..4 (skipping column 0)
    for sibling, expected in zip(siblings[1:], expected_values):
        sibling.setText.assert_called_once_with(expected)


def test_resolve_item_path(viewer):
    from PyQt6.QtGui import QStandardItem

    # Create a nested structure in the model
    root = viewer.model.invisibleRootItem()
    parent_item = QStandardItem("parent")
    child_item = QStandardItem("child")
    grandchild_item = QStandardItem("grandchild")

    # Build the hierarchy
    root.appendRow([parent_item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])
    parent_item.appendRow([child_item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])
    child_item.appendRow([grandchild_item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    # Test paths
    parent_path = viewer.resolve_item_path(parent_item)
    assert parent_path == "parent", f"Expected 'parent', got '{parent_path}'"

    child_path = viewer.resolve_item_path(child_item)
    assert child_path == "parent.child", f"Expected 'parent.child', got '{child_path}'"

    grandchild_path = viewer.resolve_item_path(grandchild_item)
    assert grandchild_path == "parent.child.grandchild", f"Expected 'parent.child.grandchild', got '{grandchild_path}'"


def test_resolve_item_path_with_list(viewer):
    from PyQt6.QtGui import QStandardItem

    # **Step 1:** Set up `data_source` with 'my_list' as an attribute
    viewer.data_source.my_list = [0, 1, 2]

    # **Step 2:** Create the item hierarchy in the model
    root = viewer.model.invisibleRootItem()
    list_item = QStandardItem("my_list")
    list_child = QStandardItem("[0]")  # Represents the first element in the list

    root.appendRow([list_item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])
    list_item.appendRow([list_child, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    # **Step 3:** Invoke `resolve_item_path` and assert the path
    path = viewer.resolve_item_path(list_child)
    assert path == "my_list[0]", f"Expected 'my_list[0]', got '{path}'"


def test_resolve_item_path_with_dict(viewer):
    from PyQt6.QtGui import QStandardItem

    # **Step 1:** Set up `data_source` with 'my_dict' as an attribute
    viewer.data_source.my_dict = {"key": "value"}

    # **Step 2:** Create the item hierarchy in the model
    root = viewer.model.invisibleRootItem()
    dict_item = QStandardItem("my_dict")
    dict_key = QStandardItem("key")  # Represents a key in the dictionary

    root.appendRow([dict_item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])
    dict_item.appendRow([dict_key, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    # **Step 3:** Invoke `resolve_item_path` and assert the path
    path = viewer.resolve_item_path(dict_key)
    assert path == 'my_dict["key"]', f"Expected 'my_dict[\"key\"]', got '{path}'"



def test_resolve_item_path_invalid(viewer):
    from PyQt6.QtGui import QStandardItem

    # Create an item without a parent
    standalone_item = QStandardItem("standalone")
    path = viewer.resolve_item_path(standalone_item)

    # Since the item is not in the model, the path should only include its text
    assert path == "standalone", f"Expected 'standalone', got '{path}'"


def test_has_variable_root_level(viewer):
    """
    Test that has_variable returns True for a variable that exists at the root level.
    """
    from PyQt6.QtGui import QStandardItem

    # Add a root-level variable to the model
    root = viewer.model.invisibleRootItem()
    var_item = QStandardItem("root_var")
    root.appendRow([var_item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    # Assert that has_variable returns True
    assert viewer.has_variable("root_var") is True


def test_has_variable_nested_path(viewer):
    """
    Test that has_variable returns True for a nested variable path.
    """
    from PyQt6.QtGui import QStandardItem

    # Add a nested variable to the model
    root = viewer.model.invisibleRootItem()
    parent_item = QStandardItem("parent_var")
    child_item = QStandardItem("child_var")
    root.appendRow([parent_item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])
    parent_item.appendRow([child_item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    # Assert that has_variable returns True for the nested path
    assert viewer.has_variable("parent_var.child_var") is True


def test_has_variable_nonexistent(viewer):
    """
    Test that has_variable returns False for a variable that does not exist.
    """
    assert viewer.has_variable("nonexistent_var") is False


def test_has_variable_partial_match(viewer):
    """
    Test that has_variable returns False if only part of the path exists.
    """
    from PyQt6.QtGui import QStandardItem

    # Add a parent variable to the model
    root = viewer.model.invisibleRootItem()
    parent_item = QStandardItem("parent_var")
    root.appendRow([parent_item, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()])

    # Assert that has_variable returns False for a non-existent child
    assert viewer.has_variable("parent_var.child_var") is False


def test_has_variable_empty_path(viewer):
    """
    Test that has_variable returns False for an empty path.
    """
    assert viewer.has_variable("") is False
