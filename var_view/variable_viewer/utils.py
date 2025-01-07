# var_view/variable_viewer/utils.py

import sys
import logging
import re
from collections.abc import Iterable

logger = logging.getLogger(__name__)


class VariableRepresentation:
    """
    Encapsulates metadata about a variable that a plugin might provide:
      - nbytes: memory usage
      - shape: shape of array-like data
      - dtype: data type
      - value_summary: a short text summarizing the contents or first elements
    """

    def __init__(self, nbytes, shape=None, dtype=None, value_summary=None):
        self.nbytes = nbytes
        self.shape = shape
        self.dtype = dtype
        self.value_summary = value_summary  # e.g. "sample=[1,2,3]..."

    def __str__(self):
        """
        Default text representation combining shape, dtype, and value summary.
        """
        parts = []
        if self.shape:
            parts.append(f"shape={self.shape}")
        if self.dtype:
            parts.append(f"dtype={self.dtype}")
        if self.value_summary:
            parts.append(str(self.value_summary))
        return ", ".join(parts) if parts else "N/A"

def infer_type_hint_general(data) -> str:
    """
    Dynamically infers a type hint for the given input data.
    Handles first-level types for nested structures, dicts, lists, sets, etc.
    For example: dict[str, int|dict], list[int|float], etc.
    """
    if isinstance(data, dict):
        # Empty dict
        if len(data) == 0:
            return "dict"
        key_types = {type(k).__name__ for k in data.keys()}
        value_types = {type(v).__name__ for v in data.values()}
        combined_keys = " | ".join(sorted(key_types))
        combined_values = " | ".join(sorted(value_types))
        return f"dict[{combined_keys}, {combined_values}]"

    elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        try:
            if len(data) == 0:
                return type(data).__name__  # e.g. "list"
        except TypeError:
            # Some iterables don't support len()
            pass
        # For a list/tuple/set, gather the unique first-level element types
        element_types = {type(element).__name__ for element in data}
        combined_type = " | ".join(sorted(element_types))
        return f"{type(data).__name__}[{combined_type}]"

    else:
        # For a non-iterable or str/bytes, just return the type name
        return type(data).__name__

def format_bytes(bytes_size):
    """
    Convert bytes to human-readable string.
    """
    try:
        units = ["B", "KB", "MB", "GB", "TB"]
        for u in units:
            if bytes_size < 1024:
                if u == "B":
                    return f"{bytes_size} B"
                return f"{bytes_size:.2f} {u}"
            bytes_size /= 1024
        return f"{bytes_size:.2f} PB"
    except Exception as e:
        logger.error(f"Error formatting bytes: {e}")
        return ""
