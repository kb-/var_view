# tests/test_drag.py
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
