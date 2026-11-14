"""FOV loading: any of jpg/png/tif/bmp, 8/16-bit, grayscale/RGB/RGBA, EXIF orientation -> 8-bit BGR uint8.

The training code read crops with cv2.imread(path) (= IMREAD_COLOR): BGR channel order, EXIF orientation applied,
alpha dropped, grayscale replicated to 3 channels, 16-bit reduced to 8-bit. For 8-bit files this loader returns
exactly the same array as cv2.imread(path) (tested); 16-bit handling is configurable (input.sixteen_bit):
cv2 = exactly cv2.imread(path) (default) | scale = round(x/257) for every format | max = scale by the image max.
"""
from pathlib import Path

import cv2
import numpy as np

try:  # silence OpenCV's own stderr messages for corrupt files; we log our own
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:  # pragma: no cover
    pass


class ImageLoadError(RuntimeError):
    pass


def load_bgr8(path, sixteen_bit="cv2"):
    """Returns (img uint8 HxWx3 BGR, info dict). Raises ImageLoadError for unreadable / unsupported files."""
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"not a file: {path}")
    try:
        # ANYDEPTH keeps 16-bit, ANYCOLOR keeps grayscale as 1 channel and drops alpha; EXIF orientation is applied
        # (it is NOT applied with IMREAD_UNCHANGED).
        im = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    except Exception as e:  # cv2 raises on some malformed files
        raise ImageLoadError(f"cv2.imread failed: {e}") from e
    if im is None or im.size == 0:
        raise ImageLoadError("unreadable or corrupt image (cv2.imread returned nothing)")
    info = {"src_dtype": str(im.dtype), "src_channels": 1 if im.ndim == 2 else int(im.shape[2])}
    if im.dtype == np.uint16:
        if sixteen_bit == "cv2":
            # exactly the training reader. Note OpenCV's own rule differs by format: TIFF -> round(x/257),
            # PNG -> x >> 8 (measured with opencv-python 5.0.0.93)
            im = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if im is None:
                raise ImageLoadError("cv2.imread(IMREAD_COLOR) failed on a 16-bit file")
        elif sixteen_bit == "scale":
            im = np.round(im.astype(np.float64) / 257.0).astype(np.uint8)
        elif sixteen_bit == "max":
            mx = int(im.max())
            im = np.zeros(im.shape, np.uint8) if mx == 0 else \
                np.clip(np.round(im.astype(np.float64) * (255.0 / mx)), 0, 255).astype(np.uint8)
        else:
            raise ImageLoadError(f"unknown sixteen_bit mode {sixteen_bit}")
    elif im.dtype != np.uint8:
        raise ImageLoadError(f"unsupported pixel type {im.dtype} (expected 8- or 16-bit integer)")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.shape[2] == 2:  # grey + alpha
        im = cv2.cvtColor(np.ascontiguousarray(im[..., 0]), cv2.COLOR_GRAY2BGR)
    elif im.shape[2] == 4:  # ANYCOLOR normally drops alpha already; keep the same rule as IMREAD_COLOR
        im = np.ascontiguousarray(im[..., :3])
    elif im.shape[2] != 3:
        raise ImageLoadError(f"unsupported channel count {im.shape[2]}")
    if im.shape[0] < 2 or im.shape[1] < 2:
        raise ImageLoadError(f"image too small: {im.shape[1]}x{im.shape[0]}")
    # grey content: a 1/2-channel file, or a colour file whose three channels are identical
    info["grey"] = info["src_channels"] in (1, 2) or bool(
        np.array_equal(im[..., 0], im[..., 1]) and np.array_equal(im[..., 1], im[..., 2]))
    return np.ascontiguousarray(im), info
