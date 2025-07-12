# Var View

Var View is a PyQt-based viewer for exploring Python variables interactively. It provides a tree view of any object and lets you inspect nested dictionaries, lists and other structures. Built-in plugins offer custom handling for common data types such as numpy arrays, OpenCV images and PyTorch tensors.

## Installation

Install the required system libraries and Python packages:

```bash
apt-get update
apt-get install -y libegl1 libxslt1.1 libxkbfile1
uv sync
uv pip install -e .
```

## Running the example

An example application is available in `main.py`:

```bash
uv run main.py
```

This launches a simple window that displays variables from a small data source. Right-click items in the tree to view additional actions such as exporting to different formats.

## Testing

Run the test suite with `pytest`. In a headless environment you may
need to provide a virtual display and use the Qt "offscreen" platform:

```bash
QT_QPA_PLATFORM=offscreen xvfb-run -a uv run -m pytest -q
```


