# src/plugins/opencv_plugin.py
import cv2

from src.variable_viewer import VariableRepresentation


def umat_formatter(value: cv2.UMat):
    try:
        mat = value.get()
        nbytes = mat.nbytes
        shape = mat.shape
        dtype = mat.dtype
        return VariableRepresentation(nbytes=nbytes, shape=shape, dtype=dtype)
    except Exception as e:
        return VariableRepresentation(nbytes=0, extra_info=f"<Error: {e}>")

def video_capture_formatter(value: cv2.VideoCapture):
    try:
        if value.isOpened():
            width = int(value.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(value.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = value.get(cv2.CAP_PROP_FPS)
            frame_count = int(value.get(cv2.CAP_PROP_FRAME_COUNT))
            extra_info = f"width={width}, height={height}, fps={fps:.2f}, frame_count={frame_count}"
        else:
            extra_info = "open=False"
        return VariableRepresentation(nbytes=0, extra_info=extra_info)
    except Exception as e:
        return VariableRepresentation(nbytes=0, extra_info=f"<Error: {e}>")

def register_handlers(register_type_handler):
    register_type_handler(cv2.UMat, umat_formatter)
    register_type_handler(cv2.VideoCapture, video_capture_formatter)
