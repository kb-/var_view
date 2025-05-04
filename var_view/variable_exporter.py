from __future__ import annotations
import os
from typing import Any, Dict, Optional, Set, TYPE_CHECKING
from PyQt6.QtWidgets import QFileDialog, QMessageBox

if TYPE_CHECKING:
    import numpy as np
    import torch
    import h5py
    import hdf5storage
    from PIL import Image


def make_matlab_compatible(
    data: Any,
    options: Optional[Any] = None,
    visited: Optional[Set[int]] = None
) -> Any:
    """
    Recursively convert `data` into forms hdf5storage can marshal under
    matlab_compatible=True.  Detects cyclic references and basic containers,
    and converts torch.Tensor → numpy.ndarray if torch is available.
    """
    if visited is None:
        visited = set()
    obj_id = id(data)
    if obj_id in visited:
        return f"<Cyclic reference to object id={obj_id}>"
    visited.add(obj_id)

    # 1) Convert torch.Tensor → NumPy if possible
    try:
        import torch, numpy as np
    except ModuleNotFoundError:
        pass
    else:
        if torch.is_tensor(data):
            return data.cpu().numpy()

    # 2) Recurse into iterable containers
    if isinstance(data, (list, tuple, set, frozenset)):
        return [make_matlab_compatible(x, options, visited) for x in data]
    if isinstance(data, dict):
        return {
            str(k): make_matlab_compatible(v, options, visited)
            for k, v in data.items()
        }

    # 3) If hdf5storage options given, attempt to find a marshaller
    if options is not None:
        marshaller = options.marshaller_collection.get_marshaller_for_type(type(data))
        if marshaller is None:
            return str(data)

    return data


class VariableExporter:
    """
    Handles exporting Python variables to various on-disk formats:
      – .npy / .npz (NumPy)
      – .csv
      – .h5 (h5py)
      – .mat (hdf5storage)
      – .png (Pillow)
      – .txt
    """

    def __init__(self, parent: Optional[Any] = None) -> None:
        self.parent = parent

    def export_variable(self, name: str, value: Any) -> None:
        """
        Prompt user for filename & extension, then dispatch to the
        appropriate save_* method. Shows a Qt file dialog and error
        messages on failure.
        """
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent,
                f"Export Variable '{name}'",
                name,
                "NumPy Array (*.npy);;CSV (*.csv);;HDF5 (*.h5);;"
                "MATLAB File (*.mat);;PNG Image (*.png);;Text File (*.txt)"
            )
            if not file_path:
                return

            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext == '.npy':
                self.save_as_npy_single(value, file_path)
            elif ext == '.npz':
                self.save_as_npz_batch({name: value}, file_path)
            elif ext == '.csv':
                self.save_as_csv(value, file_path)
            elif ext == '.h5':
                self.save_as_h5_single(name, value, file_path)
            elif ext == '.mat':
                self.save_as_mat_single(name, value, file_path)
            elif ext == '.png':
                self.save_as_png(value, file_path)
            elif ext == '.txt':
                self.save_as_txt_single(name, value, file_path)
            else:
                self._unsupported_format()
        except Exception as e:
            QMessageBox.critical(
                self.parent, "Export Error",
                f"Failed to export '{name}': {e}"
            )

    def export_variables(self, variables_dict):
        """
        Export multiple variables to the selected file format.
        """
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent,
                "Export Variables",
                "variables",
                "NumPy Archive (*.npz);;NumPy Array (*.npy);;CSV (*.csv);;HDF5 (*.h5);;MATLAB File (*.mat);;PNG Image (*.png);;Text File (*.txt)"
            )
            if not file_path:  # user cancelled
                return

            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext == ".npz":
                self.save_as_npz_batch(variables_dict, file_path)
            if ext == ".npy":
                self.save_as_npy_single(variables_dict, file_path)
            elif ext == ".csv":
                self.save_as_csv(variables_dict, file_path)
            elif ext == ".h5":
                self.save_as_h5_batch(variables_dict, file_path)
            elif ext == ".mat":
                self.save_as_mat_batch(variables_dict, file_path)
            elif ext == ".txt":
                self.save_as_txt_batch(variables_dict, file_path)
            else:
                QMessageBox.warning(
                    self.parent, "Export Error",
                    f"Unsupported format for multiple variables: {ext}"
                )
        except Exception as e:
            QMessageBox.critical(
                self.parent, "Export Error",
                f"Failed to export variables: {e}"
            )


    def _unsupported_format(self, *_) -> None:
        """Show a warning for unknown file extensions."""
        QMessageBox.warning(self.parent, "Export Error", "Unsupported format")

    # — NumPy (.npy / .npz) —————————————————————————————

    def save_as_npy_single(self, value: Any, file_path: str) -> None:
        """
        Save a single array or torch.Tensor to .npy.
        Raises RuntimeError if NumPy is missing.
        """
        try:
            import numpy as np
        except ModuleNotFoundError:
            raise RuntimeError("NumPy is required for .npy export")

        try:
            import torch
        except ModuleNotFoundError:
            torch = None  # type: ignore

        if isinstance(value, np.ndarray):
            np.save(file_path, value)
        elif torch is not None and torch.is_tensor(value):
            np.save(file_path, value.cpu().numpy())
        else:
            raise TypeError("Unsupported type for NumPy export.")

    def save_as_npz_batch(self, variables: Dict[str, Any], file_path: str) -> None:
        """
        Save multiple variables into a compressed .npz archive.
        Converts sets/frozensets to lists and torch.Tensor → numpy arrays.
        """
        try:
            import numpy as np
        except ModuleNotFoundError:
            raise RuntimeError("NumPy is required for .npz export")

        try:
            import torch
        except ModuleNotFoundError:
            torch = None  # type: ignore

        npz_dict: Dict[str, Any] = {}
        for k, v in variables.items():
            if isinstance(v, np.ndarray):
                npz_dict[k] = v
            elif torch is not None and torch.is_tensor(v):
                npz_dict[k] = v.cpu().numpy()
            elif isinstance(v, (set, frozenset)):
                npz_dict[k] = list(v)
            elif hasattr(v, '__dict__'):
                npz_dict[k] = v.__dict__
            else:
                npz_dict[k] = v
        np.savez_compressed(file_path, **npz_dict)

    # — CSV ———————————————————————————————————————————————

    def save_as_csv(self, value_or_dict: Any, file_path: str) -> None:
        """
        Export either:
          - a single array/tensor/list → to CSV via numpy.savetxt
          - a dict of 1D sequences    → to CSV via pandas DataFrame
        Requires NumPy, and pandas only if exporting a dict.
        """
        try:
            import numpy as np
        except ModuleNotFoundError:
            raise RuntimeError("NumPy is required for CSV export")

        # Helper: convert a scalar sequence/tensor to a 2D numpy array
        def _to_array(v):
            try:
                import torch
            except ModuleNotFoundError:
                torch = None  # type: ignore

            if torch is not None and torch.is_tensor(v):
                arr = v.cpu().numpy()
            elif isinstance(v, np.ndarray):
                arr = v
            elif isinstance(v, (list, tuple)):
                arr = np.array(v)
            else:
                raise TypeError(f"Unsupported type for CSV export: {type(v)}")

            # Ensure 2D for savetxt: (N,) → (N,1)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            return arr

        # --- Multi-variable case: dict → DataFrame
        if isinstance(value_or_dict, dict):
            try:
                import pandas as pd
            except ModuleNotFoundError:
                raise RuntimeError("pandas is required for CSV export of multiple variables")

            cols: Dict[str, list] = {}
            length = None
            for name, raw in value_or_dict.items():
                arr = _to_array(raw)
                if arr.shape[1] != 1:
                    raise ValueError(f"Variable '{name}' must be 1D to export together")
                col = arr.ravel().tolist()
                if length is None:
                    length = len(col)
                elif len(col) != length:
                    raise ValueError("All variables must have the same length")
                cols[name] = col

            df = pd.DataFrame(cols)
            df.to_csv(file_path, index=False)
            return

        # --- Single-variable case
        arr = _to_array(value_or_dict)
        np.savetxt(file_path, arr, delimiter=",")

    # — HDF5 (.h5) —————————————————————————————————————————

    def save_as_h5_single(self, name: str, value: Any, file_path: str) -> None:
        """
        Save one variable into an HDF5 file under dataset `name`.
        Requires h5py.
        """
        try:
            import h5py
        except ModuleNotFoundError:
            raise RuntimeError("h5py is required for .h5 export")

        import numpy as np  # always available if h5py is
        with h5py.File(file_path, 'w') as f:
            if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
                arr = value.cpu().numpy()
                f.create_dataset(name, data=arr)
            elif isinstance(value, np.ndarray):
                f.create_dataset(name, data=value)
            else:
                # fallback: store as a fixed-length string
                f.create_dataset(name, data=np.string_(str(value)))

    def save_as_h5_batch(self, variables: Dict[str, Any], file_path: str) -> None:
        """
        Save multiple variables into an HDF5 file. Unsupported types
        are placed in file attributes.
        """
        try:
            import h5py
        except ModuleNotFoundError:
            raise RuntimeError("h5py is required for .h5 export")

        import numpy as np
        with h5py.File(file_path, 'w') as f:
            for k, v in variables.items():
                if hasattr(v, 'cpu') and hasattr(v, 'numpy'):
                    data = v.cpu().numpy()
                    f.create_dataset(k, data=data)
                elif isinstance(v, np.ndarray):
                    f.create_dataset(k, data=v)
                else:
                    f.attrs[k] = str(v)

    # — MATLAB (.mat) ——————————————————————————————————————

    def save_as_mat_single(self, name: str, value: Any, file_path: str) -> None:
        """
        Save a single variable to MATLAB .mat (HDF5-based). Requires hdf5storage.
        """
        try:
            import hdf5storage
        except ModuleNotFoundError:
            raise RuntimeError("hdf5storage is required for .mat export")

        opts = hdf5storage.Options(
            matlab_compatible=True, store_python_metadata=True
        )
        data = make_matlab_compatible(value, opts)
        hdf5storage.savemat(file_path, {name: data}, format='7.3', options=opts)

    def save_as_mat_batch(self, variables: Dict[str, Any], file_path: str) -> None:
        """
        Save multiple variables to a .mat file. Converts unsupported types
        via make_matlab_compatible.
        """
        try:
            import hdf5storage
        except ModuleNotFoundError:
            raise RuntimeError("hdf5storage is required for .mat export")

        opts = hdf5storage.Options(
            matlab_compatible=True, store_python_metadata=True
        )
        data = {k: make_matlab_compatible(v, opts) for k, v in variables.items()}
        hdf5storage.savemat(file_path, data, format='7.3', options=opts)

    # — PNG ——————————————————————————————————————————————

    def save_as_png(self, value: Any, file_path: str) -> None:
        """
        Normalize a 2D or 3D NumPy array to [0,255], convert to uint8,
        and save via Pillow. Raises if shape/dtype unsupported.
        """
        try:
            import numpy as np
            from PIL import Image
        except ModuleNotFoundError:
            raise RuntimeError("NumPy and Pillow are required for PNG export")

        if not isinstance(value, np.ndarray):
            raise TypeError("PNG export only supports NumPy arrays.")

        arr = value.astype(np.float32)
        mn, mx = arr.min(), arr.max()

        if mn == mx:
            norm = np.zeros_like(arr, dtype=np.uint8)
            norm[:] = 255 if mn > 0 else 0
        else:
            norm = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

        # 2D grayscale or 3D RGB/RGBA
        if norm.ndim == 2:
            img = Image.fromarray(norm)
        elif norm.ndim == 3 and norm.shape[2] in (3, 4):
            mode = 'RGB' if norm.shape[2] == 3 else 'RGBA'
            img = Image.fromarray(norm, mode)
        else:
            raise ValueError(
                "PNG export supports shape (H,W) or (H,W,3/4) arrays."
            )

        img.save(file_path)

    # — Text (.txt) —————————————————————————————————————

    def save_as_txt_single(self, name: str, value: Any, file_path: str) -> None:
        """
        Write a header line '# Variable: {name}' then `str(value)` (or
        decoded bytes/bytearray), ensuring exactly one trailing newline.
        """
        text = f"# Variable: {name}\n"
        if isinstance(value, (bytes, bytearray)):
            text += value.decode('utf-8')
        else:
            text += str(value)
        if not text.endswith("\n"):
            text += "\n"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)

    def save_as_txt_batch(self, variables: Dict[str, Any], file_path: str) -> None:
        """
        Sequentially write multiple '# Variable: name' blocks for each
        entry in `variables`, preserving the exact newline rules.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for name, value in variables.items():
                block = f"# Variable: {name}\n"
                if isinstance(value, (bytes, bytearray)):
                    block += value.decode('utf-8')
                else:
                    block += str(value)
                if not block.endswith("\n"):
                    block += "\n"
                f.write(block)
