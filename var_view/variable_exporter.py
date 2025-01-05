# var_view/variable_exporter.py
import numpy as np
import torch
import h5py
import hdf5storage
from PIL import Image
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import os
import pickle

def make_matlab_compatible(data, options=None, visited=None):
    """
    Recursively convert data into forms hdf5storage can handle under
    matlab_compatible=True. Leaves alone data that hdf5storage can
    already marshal. Converts only truly unsupported types.
    """
    # Track already-visited objects to prevent infinite recursion
    if visited is None:
        visited = set()
    obj_id = id(data)
    if obj_id in visited:
        # Break any cycles by storing a string note about recursion
        return f"<Cyclic reference to object id={obj_id}>"
    visited.add(obj_id)

    # 1. Convert torch.Tensor → NumPy
    if torch.is_tensor(data):
        data = data.cpu().numpy()

    # 2. Recurse into containers
    if isinstance(data, (list, tuple)):
        # Convert each element, and possibly turn tuple into list
        new_list = [make_matlab_compatible(x, options, visited) for x in data]
        # If it was originally a list, keep it as a list
        # If you need to preserve "tuple vs. list", you could do so,
        # but MATLAB doesn't differentiate them anyway.
        data = new_list
    elif isinstance(data, (set, frozenset)):
        # Convert sets → list
        new_list = [make_matlab_compatible(x, options, visited) for x in data]
        data = new_list
    elif isinstance(data, dict):
        # Convert each value
        new_dict = {}
        for k, v in data.items():
            # Convert the key to string if it's not already
            # so that MATLAB doesn't choke on weird keys
            if not isinstance(k, str):
                k = str(k)
            new_dict[k] = make_matlab_compatible(v, options, visited)
        data = new_dict

    # 3. Check if hdf5storage can marshal the resulting object
    #    (That includes built-in Python types, np.ndarray, etc.)
    if options is not None:
        # Grab the marshaller for the object’s current Python type
        marshaller = options.marshaller_collection.get_marshaller_for_type(type(data))
        if marshaller is None:
            # Fallback: store the object as string
            data = str(data)

    return data

class VariableExporter:
    def __init__(self, parent=None):
        self.parent = parent  # To link to a GUI parent if needed

    def export_variable(self, name, value):
        """Export a single variable to the selected file format."""
        try:
            # Open file save dialog to let the user choose format
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self.parent,
                f"Export Variable '{name}'",
                f"{name}",
                "NumPy Array (*.npy);;CSV (*.csv);;HDF5 (*.h5);;MATLAB File (*.mat);;PNG Image (*.png);;Text File (*.txt)"
            )

            if not file_path:  # User canceled
                return

            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            # Attempt to export based on the chosen file format
            if ext == ".npy":
                self.save_as_npy_single(value, file_path)
            elif ext == ".csv":
                self.save_as_csv(value, file_path)
            elif ext == ".h5":
                self.save_as_h5_single(name, value, file_path)
            elif ext == ".mat":
                self.save_as_mat_single(name, value, file_path)
            elif ext == ".png":
                self.save_as_png(value, file_path)
            elif ext == ".txt":
                self.save_as_txt_single(name, value, file_path)
            else:
                QMessageBox.warning(self.parent, "Export Error",
                                    f"Unsupported format: {ext}")
        except Exception as e:
            QMessageBox.critical(self.parent, "Export Error",
                                 f"Failed to export '{name}': {str(e)}")

    def export_variables(self, variables_dict):
        """Export multiple variables to the selected file format."""
        try:
            # Open file save dialog to let the user choose format
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self.parent,
                f"Export Variables",
                f"variables",
                "NumPy Archive (*.npz);;HDF5 (*.h5);;MATLAB File (*.mat);;Text File (*.txt)"
            )

            if not file_path:  # User canceled
                return

            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            # Attempt to export based on the chosen file format
            if ext == ".npz":
                self.save_as_npz_batch(variables_dict, file_path)
            elif ext == ".h5":
                self.save_as_h5_batch(variables_dict, file_path)
            elif ext == ".mat":
                self.save_as_mat_batch(variables_dict, file_path)
            elif ext == ".txt":
                self.save_as_txt_batch(variables_dict, file_path)
            else:
                QMessageBox.warning(self.parent, "Export Error",
                                    f"Unsupported format for multiple variables: {ext}")
        except Exception as e:
            QMessageBox.critical(self.parent, "Export Error",
                                 f"Failed to export variables: {str(e)}")

    def save_as_npy_single(self, value, file_path):
        """Save a single variable as a .npy file."""
        if isinstance(value, np.ndarray):
            np.save(file_path, value)
        elif torch.is_tensor(value):
            np.save(file_path, value.cpu().numpy())
        else:
            raise TypeError("Unsupported type for NumPy export.")

    def save_as_npz_batch(self, variables_dict, file_path):
        """Save multiple variables into a single .npz file."""
        npz_dict = {}
        for name, value in variables_dict.items():
            if isinstance(value, np.ndarray):
                npz_dict[name] = value
            elif torch.is_tensor(value):
                npz_dict[name] = value.cpu().numpy()
            elif isinstance(value, (set, frozenset)):
                npz_dict[name] = list(value)
            elif hasattr(value, "__dict__"):
                npz_dict[name] = value.__dict__
            elif isinstance(value, (str, bytes, bytearray, int, float, bool, complex, list, tuple, dict)):
                npz_dict[name] = value
            else:
                npz_dict[name] = str(value)  # Convert unsupported types to strings
        np.savez_compressed(file_path, **npz_dict)

    def save_as_csv(self, value, file_path):
        """Save a single variable as a .csv file."""
        if isinstance(value, np.ndarray):
            np.savetxt(file_path, value, delimiter=",")
        elif torch.is_tensor(value):
            np.savetxt(file_path, value.cpu().numpy(), delimiter=",")
        else:
            raise TypeError("Unsupported type for CSV export.")

    def save_as_h5_single(self, name, value, file_path):
        """Save a single variable into a .h5 file."""
        with h5py.File(file_path, "w") as f:
            if isinstance(value, np.ndarray):
                f.create_dataset(name, data=value)
            elif torch.is_tensor(value):
                f.create_dataset(name, data=value.cpu().numpy())
            elif isinstance(value, (str, bytes, bytearray)):
                # Save strings as datasets
                f.create_dataset(name, data=np.string_(value))
            elif isinstance(value, (int, float, bool, complex)):
                f.create_dataset(name, data=value)
            elif isinstance(value, (list, tuple, dict)):
                # Save lists and dicts as HDF5 references or convert to object arrays
                # For simplicity, convert to object arrays
                f.create_dataset(name, data=np.array(value, dtype=object))
            else:
                f.attrs[name] = str(value)  # Save unsupported types as string metadata

    def save_as_h5_batch(self, variables_dict, file_path):
        """Save multiple variables into a single .h5 file."""
        with h5py.File(file_path, "w") as f:
            for name, value in variables_dict.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(name, data=value)
                elif torch.is_tensor(value):
                    f.create_dataset(name, data=value.cpu().numpy())
                elif isinstance(value, (str, bytes, bytearray)):
                    # Save strings as datasets
                    f.create_dataset(name, data=np.string_(value))
                elif isinstance(value, (int, float, bool, complex)):
                    f.create_dataset(name, data=value)
                elif isinstance(value, (list, tuple, dict)):
                    # Save lists and dicts as HDF5 references or convert to object arrays
                    # For simplicity, convert to object arrays
                    f.create_dataset(name, data=np.array(value, dtype=object))
                else:
                    f.attrs[name] = str(value)  # Save unsupported types as string metadata

    def save_as_mat_single(self, name, value, file_path):
        """
        Save a single variable into a .mat file using hdf5storage,
        converting unsupported data into forms that MATLAB can handle.
        """
        # Create Options for MATLAB compatibility
        options = hdf5storage.Options(
            matlab_compatible=True,
            store_python_metadata=True
        )

        # Recursively convert only unsupported data
        converted_value = make_matlab_compatible(value, options=options)

        try:
            hdf5storage.savemat(
                file_path,
                {name: converted_value},
                format='7.3',
                options=options
            )
        except Exception as e:
            raise ValueError(f"Failed to save '{name}' to .mat: {e}")

    def save_as_mat_batch(self, variables_dict, file_path):
        """
        Save multiple variables into a single .mat file using hdf5storage,
        converting only truly unsupported data (e.g. torch.Tensor).
        """
        options = hdf5storage.Options(
            matlab_compatible=True,
            store_python_metadata=True
        )

        # Recursively convert only unsupported data
        converted_dict = {
            var_name: make_matlab_compatible(val, options=options)
            for var_name, val in variables_dict.items()
        }

        try:
            hdf5storage.savemat(
                file_path,
                converted_dict,
                format='7.3',
                options=options
            )
        except Exception as e:
            raise ValueError(f"Failed to save variables to .mat: {e}")

    def save_as_png(self, value, file_path):
        """Save a single variable as a .png image."""
        if isinstance(value, np.ndarray):
            # Normalize values if necessary
            if value.dtype != np.uint8:
                value_min = value.min()
                value_max = value.max()
                if value_min == value_max:
                    # Avoid division by zero; output a single color
                    normalized = np.zeros_like(value, dtype=np.uint8)
                    normalized[:] = 255 if value_min > 0 else 0
                else:
                    normalized = ((value - value_min) / (value_max - value_min) * 255).astype(np.uint8)
            else:
                normalized = value  # Already uint8

            # Handle image dimensions
            if normalized.ndim == 2:  # Grayscale
                img = Image.fromarray(normalized)
            elif normalized.ndim == 3 and normalized.shape[2] in [3, 4]:  # RGB or RGBA
                img = Image.fromarray(normalized, "RGB")
            else:
                raise ValueError("PNG export supports 2D or 3D arrays with shape (H, W) or (H, W, 3/4).")

            # Save the image
            img.save(file_path)
        elif torch.is_tensor(value):
            self.save_as_png(value.cpu().numpy(), file_path)
        else:
            raise TypeError("Unsupported type for PNG export.")

    # def save_as_txt_single(self, name, value, file_path):
    #     """Save a single variable into a .txt file."""
    #     with open(file_path, "w") as f:
    #         f.write(f"# Variable: {name}\n")
    #         if isinstance(value, (list, tuple, dict)):
    #             f.write(str(value) + "\n")
    #         elif isinstance(value, np.ndarray):
    #             f.write(str(value) + "\n")
    #         elif torch.is_tensor(value):
    #             f.write(str(value.cpu().numpy()) + "\n")
    #         elif isinstance(value, (str, bytes, bytearray)):
    #             f.write(value.decode('utf-8') if isinstance(value, (bytes, bytearray)) else value + "\n")
    #         elif isinstance(value, (int, float, bool, complex)):
    #             f.write(str(value) + "\n")
    #         else:
    #             # For unsupported types, export their string representation
    #             f.write(f"{str(value)}\n")
    def save_as_txt_single(self, name, value, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Variable: {name}\n")

            if isinstance(value, (bytes, bytearray)):
                # Decode then add a newline
                f.write(value.decode("utf-8") + "\n")
            else:
                # Some fallback for other types
                f.write(str(value) + "\n")

    # def save_as_txt_batch(self, variables_dict, file_path):
    #     """Save multiple variables into a single .txt file."""
    #     with open(file_path, "w") as f:
    #         for name, value in variables_dict.items():
    #             f.write(f"# Variable: {name}\n")
    #             if isinstance(value, (list, tuple, dict)):
    #                 f.write(str(value) + "\n")
    #             elif isinstance(value, np.ndarray):
    #                 f.write(str(value) + "\n")
    #             elif torch.is_tensor(value):
    #                 f.write(str(value.cpu().numpy()) + "\n")
    #             elif isinstance(value, (str, bytes, bytearray)):
    #                 f.write(value.decode('utf-8') if isinstance(value, (bytes, bytearray)) else value + "\n")
    #             elif isinstance(value, (int, float, bool, complex)):
    #                 f.write(str(value) + "\n")
    #             else:
    #                 f.write(f"{str(value)}\n")
    def save_as_txt_batch(self, variables_dict, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for name, value in variables_dict.items():
                f.write(f"# Variable: {name}\n")
                if isinstance(value, (bytes, bytearray)):
                    f.write(value.decode("utf-8") + "\n")
                else:
                    f.write(str(value) + "\n")

