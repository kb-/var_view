# Var View

Var View is a PyQt-based viewer for exploring Python variables interactively. It
provides a tree view of any object and lets you inspect nested dictionaries,
lists and other structures. Built‑in plugins offer custom handling for common
data types such as numpy arrays, OpenCV images and PyTorch tensors. The viewer
can be embedded in any PyQt6 application so you can inspect your program state
while it is running.

## Installation

Install the required system libraries and Python packages:

```bash
pip install uv
apt-get update
apt-get install -y libegl1 libxslt1.1 libxkbfile1 xvfb
uv sync
uv pip install -e .
```

## Running the example

An example application is available in `main.py`:

```bash
uv run main.py
```

This launches a simple window that displays variables from a small data source.
Right-click items in the tree to view additional actions such as exporting to
different formats.

## Using Var View in your PyQt6 application

To embed the viewer in your own application simply import `VariableViewer` and
pass it an object whose attributes should be displayed. The viewer automatically
scans the object's attributes and presents them in a tree view.

```python
from PyQt6.QtWidgets import QApplication
from var_view import VariableViewer


class DataSource:
    def __init__(self):
        self.value = [1, 2, 3]
        self.message = "hello"


data_source = DataSource()
app = QApplication([])
viewer = VariableViewer(data_source, alias="data")
viewer.show()
app.exec()
```

The viewer can be combined with other widgets in your window and supports
built‑in plugins for common data types.

### Opening an interactive console

Call ``add_console()`` to open a Jupyter QtConsole with the data source
available inside:

```python
viewer.add_console("data")
```

This launches a separate window where you can inspect and modify the
variables interactively. Changes are reflected in the viewer when cells
finish executing.

## Testing

Run the test suite with `pytest`. In a headless environment you may
need to provide a virtual display and use the Qt "offscreen" platform.
Ensure `DISPLAY` is set and xvfb is running so PyQt can create windows:

```bash
export DISPLAY=:99
QT_QPA_PLATFORM=offscreen \
    xvfb-run --server-args="-screen 0 1280x1024x24" \
    uv run -m pytest -vv

If tests appear to hang, print the environment with `env` to confirm
that `DISPLAY` is set correctly.
```

If the full suite takes too long, run just the lightweight model tests:

```bash
uv run -m pytest tests/test_model.py -q
```


