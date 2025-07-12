# Var View

Var View is a PyQt-based viewer for exploring Python variables interactively. It provides a tree view of any object and lets you inspect nested dictionaries, lists and other structures. Built-in plugins offer custom handling for common data types such as numpy arrays, OpenCV images and PyTorch tensors.

## Installation

Use `uv` to install the package in editable mode while developing:

```bash
uv pip install -e .
```

## Running the example

An example application is available in `main.py`:

```bash
python main.py
```

This launches a simple window that displays variables from a small data source. Right-click items in the tree to view additional actions such as exporting to different formats.

## Testing

Run the test suite with `pytest`:

```bash
pytest -q
```


