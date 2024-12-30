# variableExporter.py
import numpy as np
import torch
import h5py
import scipy.io as sio
from PIL import Image
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import os


class VariableExporter:
    def __init__(self, parent=None):
        self.parent = parent  # To link to a GUI parent if needed

    def export_variable(self, name, value):
        """Export a variable to the selected file format."""
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
                self.save_as_npy(value, file_path)
            elif ext == ".csv":
                self.save_as_csv(value, file_path)
            elif ext == ".h5":
                self.save_as_h5(name, value, file_path)
            elif ext == ".mat":
                self.save_as_mat(name, value, file_path)
            elif ext == ".png":
                self.save_as_png(value, file_path)
            elif ext == ".txt":
                self.save_as_txt(name, value, file_path)
            else:
                QMessageBox.warning(self.parent, "Export Error",
                                    f"Unsupported format: {ext}")
        except Exception as e:
            QMessageBox.critical(self.parent, "Export Error",
                                 f"Failed to export {name}: {str(e)}")

    def save_as_npy(self, value, file_path):
        if isinstance(value, np.ndarray):
            np.save(file_path, value)
        elif torch.is_tensor(value):
            np.save(file_path, value.cpu().numpy())
        else:
            raise TypeError("Unsupported type for NumPy export.")

    def save_as_csv(self, value, file_path):
        if isinstance(value, np.ndarray):
            np.savetxt(file_path, value, delimiter=",")
        elif torch.is_tensor(value):
            np.savetxt(file_path, value.cpu().numpy(), delimiter=",")
        else:
            raise TypeError("Unsupported type for CSV export.")

    def save_as_h5(self, name, value, file_path):
        with h5py.File(file_path, "w") as f:
            if isinstance(value, np.ndarray):
                f.create_dataset(name, data=value)
            elif torch.is_tensor(value):
                f.create_dataset(name, data=value.cpu().numpy())
            else:
                f.attrs[name] = str(value)  # Save unsupported types as string metadata

    def save_as_mat(self, name, value, file_path):
        """
        Save the given variable to a MATLAB file while preserving its name.
        """
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()  # Convert PyTorch tensor to NumPy array

        try:
            # Save the variable using its name
            sio.savemat(file_path, {name: value})
        except Exception as e:
            raise ValueError(f"Failed to save {name} to .mat: {e}")

    def save_as_png(self, value, file_path):
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
                    normalized = ((value - value_min) / (
                                value_max - value_min) * 255).astype(np.uint8)
            else:
                normalized = value  # Already uint8

            # Handle image dimensions
            if normalized.ndim == 2:  # Grayscale
                img = Image.fromarray(normalized)
            elif normalized.ndim == 3 and normalized.shape[2] in [3, 4]:  # RGB or RGBA
                img = Image.fromarray(normalized, "RGB")
            else:
                raise ValueError(
                    "PNG export supports 2D or 3D arrays with shape (H, W) or (H, W, 3/4).")

            # Save the image
            img.save(file_path)
        elif torch.is_tensor(value):
            self.save_as_png(value.cpu().numpy(), file_path)
        else:
            raise TypeError("Unsupported type for PNG export.")

    def save_as_txt(self, name, value, file_path):
        with open(file_path, "w") as f:
            if isinstance(value, (list, tuple, dict)):
                f.write(str(value))  # Export structured data as string
            elif isinstance(value, np.ndarray):
                np.savetxt(f, value, fmt="%.5f")  # Export numerical arrays
            elif torch.is_tensor(value):
                np.savetxt(f, value.cpu().numpy(), fmt="%.5f")  # Export PyTorch tensors
            elif isinstance(value, str):
                f.write(value)  # Write strings directly
            else:
                # For unsupported types, export their string representation
                f.write(f"# Exported variable: {name}\n{str(value)}")
