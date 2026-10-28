import math
import warnings
import numpy as np


def default_K(width=640, height=480, focal=500.0):
    return np.array([[focal, 0.0, width / 2.0],
                     [0.0, focal, height / 2.0],
                     [0.0, 0.0, 1.0]])


def K_from_exif(path):
    import exifread
    with open(path, "rb") as f:
        tags = exifread.process_file(f)
    try:
        w = int(tags["EXIF ExifImageWidth"].values[0])
        h = int(tags["EXIF ExifImageLength"].values[0])
        f35 = float(tags["EXIF FocalLengthIn35mmFilm"].values[0])
    except KeyError:
        return None
    diag_35 = math.hypot(36.0, 24.0)
    diag_px = math.hypot(w, h)
    f_px = f35 / diag_35 * diag_px
    cx = w / 2
    cy = h / 2
    K = np.array([
        [f_px, 0.0, cx],
        [0.0, f_px, cy],
        [0.0, 0.0, 1.0],
    ])
    return K


def get_K(path=None, width=640, height=480, focal=500.0):
    if path is not None:
        K = K_from_exif(path)
        if K is not None:
            return K
    return default_K(width, height, focal)