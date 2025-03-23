# tests/test_drag.py
import re

import pytest
import pyautogui  # Install with: pip install pyautogui
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt, QPoint
from var_view.variable_viewer.viewer import VariableViewer

# Import the data source from main.py (adjust if needed)
from main import AppDataSource


@pytest.fixture
def viewer(qtbot):
    # Create an instance of your data source and viewer
    data_source = AppDataSource()
    viewer = VariableViewer(data_source, alias="c")
    viewer.show()
    qtbot.addWidget(viewer)
    qtbot.waitExposed(viewer)
    return viewer


def test_dragging_each_root_entry(qtbot, viewer):
    """
    For each root entry in the viewer's tree, simulate a drag operation
    and verify that the mime data text is a safe string (e.g. starts with "c.").

    At the beginning, physically move the mouse pointer over the viewer window
    using PyAutoGUI to help "unstick" pytest-qt on Windows.
    """

    tree_view = viewer.tree_view
    model = viewer.model
    root_item = model.invisibleRootItem()
    num_rows = root_item.rowCount()
    qtbot.wait(500)  # Wait a moment after the physical move

    # Move physical mouse pointer into the viewer window.
    viewer_center = viewer.mapToGlobal(viewer.rect().center())
    # qtbot.wait(500)  # Wait a moment after the physical move
    pyautogui.moveTo(viewer_center.x(), viewer_center.y())
    # qtbot.wait(500)  # Wait a moment after the physical move

    for row in range(num_rows):
        index = model.index(row, 0)
        # Retrieve the mime data directly from the model
        mime_data = model.mimeData([index])
        drag_text = mime_data.text()
        print("drag_text", drag_text)
        assert drag_text.startswith(
            "c."), f"Row {row} drag text does not start with 'c.': {drag_text}"

        # Now simulate the drag gesture on the tree view:
        # Get the visual rectangle of the item and use its center as start.
        rect = tree_view.visualRect(index)
        start_pos = rect.center()
        # Offset by (20,20) to exceed the drag threshold.
        end_pos = start_pos + QPoint(200, 200)

        # Simulate mouse press, move, and release.
        pyautogui.moveTo(viewer_center.x(), viewer_center.y())
        QTest.mousePress(tree_view.viewport(), Qt.MouseButton.LeftButton, pos=start_pos)
        QTest.mouseMove(tree_view.viewport(), end_pos)
        QTest.mouseRelease(tree_view.viewport(), Qt.MouseButton.LeftButton, pos=end_pos)
        pyautogui.moveTo(viewer_center.x()+10, viewer_center.y())

        # If no exception is raised, the drag operation for this root entry succeeded.

def test_dragging_child_entry(qtbot, viewer):
    """
    Expand the tree down to a nested child and simulate a drag on it.
    Verify that the mime data text equals:
    c.complex_nested_dict['level1']['level2']['level3'][0]
    """
    tree_view = viewer.tree_view
    model = viewer.model

    # Find the root "complex_nested_dict" item.
    root_item = model.invisibleRootItem()
    target_index = None

    # Move physical mouse pointer into the viewer window.
    viewer_center = viewer.mapToGlobal(viewer.rect().center())
    # qtbot.wait(500)  # Wait a moment after the physical move
    pyautogui.moveTo(viewer_center.x(), viewer_center.y())
    # qtbot.wait(500)  # Wait a moment after the physical move

    for row in range(root_item.rowCount()):
        item = root_item.child(row, 0)
        if item.text() == "complex_nested_dict":
            target_index = model.index(row, 0)
            break
    assert target_index is not None, "Root entry 'complex_nested_dict' not found."

    # Expand "complex_nested_dict"
    tree_view.expand(target_index)
    qtbot.wait(500)

    # Find and expand child "level1"
    level1_index = None
    root_item = model.itemFromIndex(target_index)
    for row in range(root_item.rowCount()):
        item = root_item.child(row, 0)
        if item.text() == "level1":
            level1_index = model.index(row, 0, target_index)
            break
    assert level1_index is not None, "Child 'level1' not found."
    tree_view.expand(level1_index)
    qtbot.wait(500)

    # Find and expand child "level2"
    level1_item = model.itemFromIndex(level1_index)
    level2_index = None
    for row in range(level1_item.rowCount()):
        item = level1_item.child(row, 0)
        if item.text() == "level2":
            level2_index = model.index(row, 0, level1_index)
            break
    assert level2_index is not None, "Child 'level2' not found."
    tree_view.expand(level2_index)
    qtbot.wait(500)

    # Find and expand child "level3"
    level2_item = model.itemFromIndex(level2_index)
    level3_index = None
    for row in range(level2_item.rowCount()):
        item = level2_item.child(row, 0)
        if item.text() == "level3":
            level3_index = model.index(row, 0, level2_index)
            break
    assert level3_index is not None, "Child 'level3' not found."
    tree_view.expand(level3_index)
    qtbot.wait(500)

    # Finally, find the first element of the list inside level3.
    level3_item = model.itemFromIndex(level3_index)
    child_index = None
    for row in range(level3_item.rowCount()):
        child = level3_item.child(row, 0)
        if child.text() == "[0]":
            child_index = model.index(row, 0, level3_index)
            break
    assert child_index is not None, "Child '[0]' not found in 'level3'."

    # Retrieve the mime data for this child
    mime_data = model.mimeData([child_index])
    drag_text = mime_data.text()
    print("drag_text", drag_text)
    expected = "c.complex_nested_dict['level1']['level2']['level3'][0]"
    assert drag_text == expected, f"Expected: {expected}, got: {drag_text}"

    # Optionally, simulate a drag gesture for extra confidence.
    rect = tree_view.visualRect(child_index)
    start_pos = rect.center()
    end_pos = start_pos + QPoint(50, 50)
    pyautogui.moveTo(viewer_center.x(), viewer_center.y())
    QTest.mousePress(tree_view.viewport(), Qt.MouseButton.LeftButton, pos=start_pos)
    QTest.mouseMove(tree_view.viewport(), end_pos)
    QTest.mouseRelease(tree_view.viewport(), Qt.MouseButton.LeftButton, pos=end_pos)
    pyautogui.moveTo(viewer_center.x() + 10, viewer_center.y())


import re
from PyQt6.QtTest import QTest
from PyQt6.QtCore import QPoint


def expand_tree_to_path(viewer, safe_path):
    """
    Given a safe path string (e.g. "c.list_obj[0].name"),
    force the tree to expand along that path and return the final QStandardItem.
    This function strips the alias ("c.") and tokenizes the remainder.
    For each token, it expands the current item (forcing lazy loading if needed)
    and searches its children for a matching text.
    """
    alias = "c."
    if not safe_path.startswith(alias):
        raise ValueError("Safe path must start with 'c.'")
    remainder = safe_path[len(alias):]

    # Tokenize: tokens are sequences of word characters or bracketed parts.
    tokens = re.findall(r'\w+|\[[^\]]+\]', remainder)
    # For example, "list_obj[0].name" yields: ["list_obj", "[0]", "name"]

    current_item = viewer.model.invisibleRootItem()
    for token in tokens:
        # Expand the current item to trigger lazy loading.
        idx = current_item.index()
        viewer.tree_view.expand(idx)
        # If the item has a "Loading..." placeholder, trigger expansion.
        if current_item.rowCount() > 0:
            first_child = current_item.child(0, 0)
            if first_child.text() == "Loading...":
                viewer.handle_expand(idx)
        # Wait a bit for children to load.
        QTest.qWait(300)

        found = False
        for row in range(current_item.rowCount()):
            child = current_item.child(row, 0)
            # For tokens that are bracketed (e.g. "[0]"), compare directly.
            if token.startswith('['):
                if child.text() == token:
                    current_item = child
                    found = True
                    break
            else:
                # Otherwise, compare the token with the child text.
                if child.text() == token:
                    current_item = child
                    found = True
                    break
        if not found:
            return None
    return current_item


@pytest.mark.parametrize("expected_path", [
    "c.list_obj[0].name",
    "c.complex_nested_dict['level1']['level2']['level3'][0]",
    "c.nested_dict['b']['d']",
    "c.cyclic_ref['self']['self']['self']",
    "c.object_key_dict[\"CustomKey(id=2, description='Second Key')\"][\'nested\'][0]"
])
def test_dragging_children_by_path(qtbot, viewer, expected_path):
    """
    For each expected safe drag path, force expansion of the tree
    so that the item is loaded, then verify its mime data text equals the expected path.
    """
    target_item = expand_tree_to_path(viewer, expected_path)
    assert target_item is not None, f"Could not expand tree to safe path '{expected_path}'"

    index = target_item.index()
    mime_data = viewer.model.mimeData([index])
    drag_text = mime_data.text()
    print("drag_text", drag_text)
    assert drag_text == expected_path, f"Expected: {expected_path}, got: {drag_text}"

    # Optionally, simulate a drag gesture.
    tree_view = viewer.tree_view
    rect = tree_view.visualRect(index)
    start_pos = rect.center()
    end_pos = start_pos + QPoint(50, 50)
    viewer_center = viewer.mapToGlobal(viewer.rect().center())
    pyautogui.moveTo(int(viewer_center.x()), int(viewer_center.y()))
    QTest.mousePress(tree_view.viewport(), Qt.MouseButton.LeftButton, pos=start_pos)
    QTest.mouseMove(tree_view.viewport(), end_pos)
    QTest.mouseRelease(tree_view.viewport(), Qt.MouseButton.LeftButton, pos=end_pos)
    pyautogui.moveTo(viewer_center.x() + 10, viewer_center.y())
