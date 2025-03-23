# tests/test_paginated_viewer.py
import pytest
from PyQt6.QtCore import Qt
from variable_viewer.paginated_viewer import PaginatedVariableViewer


# Create a dummy data source with a long list attribute.
class DummyDataSource:
    def __init__(self):
        # A list with 300 integer items.
        self.test_list = list(range(300))


@pytest.fixture
def dummy_data_source():
    return DummyDataSource()


@pytest.fixture
def viewer(qtbot, dummy_data_source):
    # Instantiate the paginated viewer with our dummy data.
    v = PaginatedVariableViewer(dummy_data_source, "dummy", "var_view/plugins")
    qtbot.addWidget(v)
    v.show()
    return v


def find_top_level_item_by_text(root, text):
    """Helper to find a top-level item in the viewer's model by its text."""
    for row in range(root.rowCount()):
        item = root.child(row, 0)
        if item.text() == text:
            return item
    return None


def test_paginated_list_initial_load(viewer, qtbot):
    # Get the root item from the viewer's model.
    root = viewer.model.invisibleRootItem()

    # Look for the 'test_list' attribute in the top-level items.
    test_list_item = find_top_level_item_by_text(root, "test_list")
    assert test_list_item is not None, "test_list attribute not found in the viewer."

    # Expand the test_list node to trigger lazy loading.
    index = test_list_item.index()
    viewer.tree_view.expand(index)

    # Wait until children are loaded (the first batch should be 100 items plus a Load More node).
    qtbot.waitUntil(lambda: test_list_item.rowCount() > 0, timeout=3000)

    # Get texts for all children of test_list.
    children_texts = [test_list_item.child(i, 0).text() for i in
                      range(test_list_item.rowCount())]
    # We expect a "Load More" node to appear because 300 > 100.
    assert any("Load More" in txt for txt in
               children_texts), "Load More node not found after initial expansion."


def test_paginated_list_load_more(viewer, qtbot):
    # Get the root item.
    root = viewer.model.invisibleRootItem()
    test_list_item = find_top_level_item_by_text(root, "test_list")
    assert test_list_item is not None, "test_list attribute not found in the viewer."

    # Expand test_list node.
    index = test_list_item.index()
    viewer.tree_view.expand(index)
    qtbot.waitUntil(lambda: test_list_item.rowCount() > 0, timeout=3000)

    # Identify the "Load More" node among the children.
    load_more_item = None
    for i in range(test_list_item.rowCount()):
        child_text = test_list_item.child(i, 0).text()
        if "Load More" in child_text:
            load_more_item = test_list_item.child(i, 0)
            break
    assert load_more_item is not None, "Load More node not found before triggering load more."

    # Record the current number of children.
    initial_count = test_list_item.rowCount()

    # Instead of simulating a mouse double-click, call the double-click handler directly.
    lm_index = load_more_item.index()
    viewer.handle_double_click(lm_index)

    # Wait until the number of children increases.
    qtbot.waitUntil(lambda: test_list_item.rowCount() > initial_count, timeout=3000)

    new_count = test_list_item.rowCount()
    assert new_count > initial_count, "Loading more did not increase the number of children."
