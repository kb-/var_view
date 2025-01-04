# src/plugins/opencv_plugin.py
import cv2

from src.variable_viewer import VariableRepresentation


def umat_formatter(value: cv2.UMat):
    try:
        mat = value.get()
        shape = mat.shape
        dtype = mat.dtype
        sample = mat.flatten()[:5].tolist()  # First 5 elements
        return VariableRepresentation(
            nbytes=mat.nbytes, shape=shape, dtype=dtype, value_summary=f"sample={sample}..."
        )
    except Exception as e:
        return VariableRepresentation(nbytes=0, value_summary=f"<Error: {e}>")


def video_capture_formatter(value: cv2.VideoCapture):
    try:
        if value.isOpened():
            width = value.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = value.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = value.get(cv2.CAP_PROP_FPS)
            frame_count = value.get(cv2.CAP_PROP_FRAME_COUNT)
            return VariableRepresentation(
                nbytes=0,
                value_summary=(
                    f"open=True, width={int(width)}, height={int(height)}, "
                    f"fps={fps:.2f}, frame_count={int(frame_count)}"
                ),
            )
        else:
            return VariableRepresentation(nbytes=0, value_summary="open=False")
    except Exception as e:
        return VariableRepresentation(nbytes=0, value_summary=f"<Error: {e}>")


def register_handlers(register_type_handler):
    register_type_handler(cv2.UMat, umat_formatter)
    register_type_handler(cv2.VideoCapture, video_capture_formatter)
