# basic preprocessing for loading a text image.
import skimage.io as img_io
import skimage.color as img_color
from skimage.transform import resize
import numpy as np
from PIL import Image

def load_image(path: str,
               *,                  # all args after this are keyword‑only
               ensure_black_bg: bool = True,
               bg_thresh: float = .55) -> np.ndarray:
    """Load one image as a grayscale float array in [0, 1].

    Args:
        path (str): Filesystem path to the source image.
        ensure_black_bg (bool): If True, invert only when background appears
            bright; if False, always invert.
        bg_thresh (float): Brightness threshold used by adaptive inversion.

    Returns:
        np.ndarray: Grayscale image array with shape (H, W) and dtype float32.
    """
    img = img_io.imread(path)
    if img.ndim == 3:
        img = img_color.rgb2gray(img)
    # img = img.astype(np.float32) / 255.0            # → [0,1]
    img = img.astype(np.float32)
    if img.dtype == np.uint8 or img.max() > 1.01:
        img = img / 255.0

    if ensure_black_bg:
        bg_est = np.median(img)
        if bg_est > bg_thresh:                      # bright paper → invert
            img = 1.0 - img
    else:                                           # legacy behaviour
        img = 1.0 - img
    return img


def preprocess(img, input_size, border_size=8):
    """Resize and pad an image to a fixed output shape.

    Args:
        img (np.ndarray): Input image with shape (H, W).
        input_size (tuple[int, int]): Target output size as (H_out, W_out).
        border_size (int): Minimum padding applied on each image side.

    Returns:
        np.ndarray: Preprocessed image with shape (H_out, W_out).
    """

    h_target, w_target = input_size

    n_height = min(h_target - 2 * border_size, img.shape[0])

    scale = n_height / img.shape[0]
    n_width = min(w_target - 2 * border_size, int(scale * img.shape[1]))

    img = resize(image=img, output_shape=(n_height, n_width)).astype(np.float32)

    # symmetric padding to input_size
    pad_v_extra = h_target - n_height - 2 * border_size
    pad_h_extra = w_target - n_width - 2 * border_size
    top = border_size + pad_v_extra // 2
    bottom = border_size + pad_v_extra - pad_v_extra // 2
    left = border_size + pad_h_extra // 2
    right = border_size + pad_h_extra - pad_h_extra // 2
    img = np.pad(img, ((top, bottom), (left, right)), mode='median')

    return img


def is_image_corrupted(path: str, fast: bool = True) -> bool:
    """
    Check if an image file appears corrupted or unreadable.

    Args:
        path (str): Filesystem path to the image to test.
        fast (bool): If True, validate headers/structure via PIL.Image.verify()
            without fully decoding pixels. If False, fully decodes the image
            via PIL.Image.open(...).load() which is slower but more robust.

    Returns:
        bool: True if the image seems corrupted/unreadable, False otherwise.
    """
    try:
        with Image.open(path) as im:
            if fast:
                # Header/structure validation without full decode
                im.verify()
            else:
                # Force full decode to catch deeper corruption
                im.load()
        return False
    except Exception:
        return True
