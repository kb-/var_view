import re
import pytest
import pyautogui  # pip install pyautogui
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt, QPoint
from var_view.variable_viewer.paginated_viewer import PaginatedVariableViewer as VariableViewer
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
    and verify that the mime data text is a safe string (e.g., starts with "c.").
    """
    tree_view = viewer.tree_view
    model = viewer.model
    root_item = model.invisibleRootItem()
    num_rows = root_item.rowCount()
    QTest.qWait(500)  # wait a moment

    # Move physical mouse pointer into the viewer window.
    viewer_center = viewer.mapToGlobal(viewer.rect().center())
    pyautogui.moveTo(viewer_center.x(), viewer_center.y())

    for row in range(num_rows):
        index = model.index(row, 0)
        mime_data = model.mimeData([index])
        drag_text = mime_data.text()
        print("drag_text", drag_text)
        assert drag_text.startswith("c."), f"Row {row} drag text does not start with 'c.': {drag_text}"

        # Simulate drag gesture on the tree view.
        rect = tree_view.visualRect(index)
        start_pos = rect.center()
        end_pos = start_pos + QPoint(200, 200)
        pyautogui.moveTo(viewer_center.x(), viewer_center.y())
        QTest.mousePress(tree_view.viewport(), Qt.MouseButton.LeftButton, pos=start_pos)
        QTest.mouseMove(tree_view.viewport(), end_pos)
        QTest.mouseRelease(tree_view.viewport(), Qt.MouseButton.LeftButton, pos=end_pos)
        pyautogui.moveTo(viewer_center.x() + 10, viewer_center.y())


def force_expand_along_tokens(viewer, tokens):
    """
    Given a list of tokens (e.g., ["list_obj", "[0]", "name"]),
    force expansion of the tree along these tokens.
    Returns the final QStandardItem if found, else None.
    """
    current_item = viewer.model.invisibleRootItem()
    for token in tokens:
        # Expand current item to trigger lazy loading.
        idx = current_item.index()
        viewer.tree_view.expand(idx)
        # If there's a "Loading..." placeholder, call handle_expand.
        if current_item.rowCount() > 0:
            first_child = current_item.child(0, 0)
            if first_child and first_child.text() == "Loading...":
                viewer.handle_expand(idx)
        QTest.qWait(30)  # allow children to load

        found = False
        # If token is bracketed, strip brackets and quotes for comparison.
        if token.startswith('[') and token.endswith(']'):
            token_cmp = token.strip("[]'\"")
        else:
            token_cmp = token

        # print(f"Expanding '{current_item.text()}' for token '{token}' (comparing as '{token_cmp}')...")
        for row in range(current_item.rowCount()):
            child = current_item.child(row, 0)
            # If child text is bracketed, normalize it too.
            if child.text().startswith('[') and child.text().endswith(']'):
                child_cmp = child.text().strip("[]'\"")
            else:
                child_cmp = child.text()
            # print(f"  child -> {child.text()} (normalized: '{child_cmp}')")
            if child_cmp == token_cmp:
                current_item = child
                found = True
                break
        if not found:
            return None
    return current_item


def expand_tree_to_path(viewer, safe_path):
    """
    Given a safe path string (e.g., "c.list_obj[0].name"),
    strip the alias ("c."), tokenize the remainder, and force expand each node.
    Tokens are defined as sequences of word characters or bracketed parts.
    Returns the final QStandardItem if found, else None.
    """
    alias = "c."
    if not safe_path.startswith(alias):
        raise ValueError("Safe path must start with 'c.'")
    remainder = safe_path[len(alias):]
    tokens = re.findall(r'\w+|\[[^\]]+\]', remainder)
    # For example, "list_obj[0].name" → ["list_obj", "[0]", "name"]
    return force_expand_along_tokens(viewer, tokens)


def collect_safe_paths(item):
    """Recursively collects safe drag paths from item and its descendants."""
    paths = []
    safe = item.data(Qt.ItemDataRole.UserRole + 3)
    if safe:
        paths.append(safe)
    for row in range(item.rowCount()):
        child = item.child(row, 0)
        paths.extend(collect_safe_paths(child))
    return paths


@pytest.mark.parametrize("expected_path", [
    "c.list_obj[0].name",
    "c.complex_nested_dict['level1']['level2']['level3'][0]",
    "c.nested_dict['b']['d']",
    "c.cyclic_ref['self']['self']['self']",
    "c.object_key_dict[\"CustomKey(id=2, description='Second Key')\"][\'nested\'][0]"
])
def test_dragging_children_by_path(qtbot, viewer, expected_path):
    """
    For each expected safe drag path, force expansion of the tree along that path
    and then verify that the mime data text for the final item equals the expected path.
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
