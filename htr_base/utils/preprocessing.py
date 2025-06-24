# basic preprocessing for loading a text image.
import skimage.io as img_io
import skimage.color as img_color
from skimage.transform import resize
import numpy as np

def load_image(path: str,
               *,                  # all args after this are keyword‑only
               ensure_black_bg: bool = True,
               bg_thresh: float = .55) -> np.ndarray:
    """
    Read *path* and return a float32 array in [0,1] whose background is black
    and foreground (text) is white.

    Parameters
    ----------
    ensure_black_bg : bool, default True
        If False, keep the legacy unconditional inversion.
    bg_thresh : float, default 0.55
        When `ensure_black_bg` is True we compute the **median** gray value of
        the image (after rgb→gray, before any inversion).  If the median is
        *brighter* than `bg_thresh` we invert, otherwise we keep the pixel
        values as-is.
    """
    img = img_io.imread(path)
    if img.ndim == 3:
        img = img_color.rgb2gray(img)
    img = img.astype(np.float32) / 255.0            # → [0,1]

    if ensure_black_bg:
        bg_est = np.median(img)
        if bg_est > bg_thresh:                      # bright paper → invert
            img = 1.0 - img
    else:                                           # legacy behaviour
        img = 1.0 - img
    return img


def preprocess(img, input_size, border_size=8):
    """Resize ``img`` and symmetrically pad it to ``input_size``."""

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
