# var_view/built_in_plugins/opencv_plugin.py

import cv2

from var_view.plugin_base import PluginBase
from var_view.variable_viewer import VariableRepresentation


class OpenCVPlugin(PluginBase):
    """
    Plugin to handle OpenCV objects like cv2.UMat and cv2.VideoCapture.
    """

    def register_handlers(self, register_type_handler):
        register_type_handler(cv2.UMat, self.umat_formatter)
        register_type_handler(cv2.VideoCapture, self.video_capture_formatter)

    def umat_formatter(self, value: cv2.UMat) -> VariableRepresentation:
        try:
            mat = value.get()
            shape = mat.shape
            dtype = str(mat.dtype)
            sample = mat.flatten()[:5].tolist()  # First 5 elements
            return VariableRepresentation(
                nbytes=mat.nbytes,
                shape=shape,
                dtype=dtype,
                value_summary=f"{sample}... dtype={dtype}"
            )
        except Exception as e:
            return VariableRepresentation(nbytes=0, value_summary=f"<Error: {e}>")

    def video_capture_formatter(self, value: cv2.VideoCapture) -> VariableRepresentation:
        try:
            if value.isOpened():
                width = value.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = value.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = value.get(cv2.CAP_PROP_FPS)
                frame_count = value.get(cv2.CAP_PROP_FRAME_COUNT)
                return VariableRepresentation(
                    nbytes=0,
                    shape=None,  # VideoCapture doesn't have a shape
                    dtype=None,  # No dtype applicable
                    value_summary=(
                        f"open=True, width={int(width)}, height={int(height)}, "
                        f"fps={fps:.2f}, frame_count={int(frame_count)}"
                    ),
                )
            else:
                return VariableRepresentation(nbytes=0, value_summary="open=False")
        except Exception as e:
            return VariableRepresentation(nbytes=0, value_summary=f"<Error: {e}>")
