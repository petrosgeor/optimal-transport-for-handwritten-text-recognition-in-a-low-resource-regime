# basic preprocessing for loading a text image.
import skimage.io as img_io
import skimage.color as img_color
from skimage.transform import resize
import numpy as np

def load_image(image_path):

    # read the image
    image = img_io.imread(image_path)

    # convert to grayscale skimage
    if len(image.shape) == 3:
        image = img_color.rgb2gray(image)
    
    # normalize the image
    image = 1 - image / 255.

    return image


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
