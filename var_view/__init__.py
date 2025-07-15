"""Public interface for the var_view package.

The :class:`~var_view.variable_viewer.viewer.VariableViewer` widget can be used
to inspect Python objects in a PyQt6 application.

Example
-------
>>> from PyQt6.QtWidgets import QApplication
>>> from var_view import VariableViewer
>>> app = QApplication([])
>>> viewer = VariableViewer(object(), alias="obj")
>>> viewer.show()
>>> viewer.add_console("obj")

"""

from .variable_viewer.viewer import VariableViewer

__all__ = ["VariableViewer"]
