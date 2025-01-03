# opencv_plugin.py
import cv2


def umat_formatter(value: cv2.UMat):
    """
    Format a cv2.UMat object for display.
    """
    try:
        # Get the shape and dtype from the underlying numpy array
        mat = value.get()
        shape = mat.shape
        dtype = mat.dtype
        return f"cv2.UMat: shape={shape}, dtype={dtype}"
    except Exception as e:
        return f"cv2.UMat: <Error: {e}>"


def video_capture_formatter(value: cv2.VideoCapture):
    """
    Format a cv2.VideoCapture object for display.
    """
    try:
        # Extract properties if the VideoCapture object is open
        if value.isOpened():
            width = value.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = value.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = value.get(cv2.CAP_PROP_FPS)
            frame_count = value.get(cv2.CAP_PROP_FRAME_COUNT)
            return (f"cv2.VideoCapture: open=True, width={int(width)}, "
                    f"height={int(height)}, fps={fps:.2f}, frame_count={int(frame_count)}")
        else:
            return "cv2.VideoCapture: open=False"
    except Exception as e:
        return f"cv2.VideoCapture: <Error: {e}>"


def register_handlers(register_type_handler):
    """
    Register handlers for OpenCV types.
    """
    register_type_handler(cv2.UMat, umat_formatter)
    register_type_handler(cv2.VideoCapture, video_capture_formatter)
