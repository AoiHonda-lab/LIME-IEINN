import numpy as np
from skimage.segmentation import quickshift
from skimage.color import gray2rgb

def exponential_kernel(distances, kernel_width=1.0):
    return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

def segment_image(image, kernel_size=4, max_dist=200, ratio=0.2):
    if len(image.shape) == 2:
        image = gray2rgb(image)
    segments = quickshift(image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
    return image, segments

def segment_image_grid(image, grid_size=4):
    """
    画像を grid_size x grid_size のグリッドで分割し、セグメントラベルを返す。
    """
    height, width = image.shape[:2]
    segments = np.zeros((height, width), dtype=int)

    h_step = height // grid_size
    w_step = width // grid_size

    label = 0
    for i in range(grid_size):
        for j in range(grid_size):
            h_start = i * h_step
            h_end = (i + 1) * h_step if i < grid_size - 1 else height
            w_start = j * w_step
            w_end = (j + 1) * w_step if j < grid_size - 1 else width

            segments[h_start:h_end, w_start:w_end] = label
            label += 1

    return image, segments