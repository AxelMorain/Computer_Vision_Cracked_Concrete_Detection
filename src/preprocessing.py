import numpy as np
import skimage as ski


def image_manipulation_take_1_0(image: np.ndarray) -> np.ndarray:
    """Preprocess an RGB image using global (Otsu) thresholding.

    Extracts the red channel, applies CLAHE contrast enhancement,
    then binarizes with a global Otsu threshold.

    Args:
        image: RGB image array of shape (H, W, 3).
    Returns:
        Binary boolean array of shape (H, W).
    """
    red = image[:, :, 0].astype(np.float32) / 255.0
    enhanced = ski.exposure.equalize_adapthist(red)
    return enhanced > ski.filters.threshold_otsu(enhanced)


def image_manipulation_take_1_1(image: np.ndarray) -> np.ndarray:
    """Preprocess an RGB image using local thresholding.

    Extracts the red channel, applies CLAHE contrast enhancement,
    then binarizes with a local (block-based) threshold.
    Preserves faint cracks that global thresholding loses.

    Args:
        image: RGB image array of shape (H, W, 3).
    Returns:
        Binary boolean array of shape (H, W).
    """
    red = image[:, :, 0].astype(np.float32) / 255.0
    enhanced = ski.exposure.equalize_adapthist(red)
    local_thresh = ski.filters.threshold_local(enhanced, block_size=41)
    return enhanced > local_thresh


def image_manipulation_take_2(image: np.ndarray) -> np.ndarray:
    """Preprocess + augment an RGB image with 4 mirror flips.

    Applies CLAHE and global Otsu threshold, then produces 5 variants:
    original + vertical flip + horizontal flip + main diagonal + anti-diagonal.
    Increases dataset size 5x and was the key step to go from 80% → 100% accuracy.

    Args:
        image: RGB image array of shape (H, W, 3).
    Returns:
        Boolean array of shape (5, H, W).
    """
    red = image[:, :, 0].astype(np.float32) / 255.0
    enhanced = ski.exposure.equalize_adapthist(red)

    vertical   = np.flipud(enhanced)       # (i,j) → (N-1-i, j)
    horizontal = np.fliplr(enhanced)       # (i,j) → (i, N-1-j)
    main_diag  = enhanced.T               # (i,j) → (j, i)
    anti_diag  = enhanced[::-1, ::-1].T   # (i,j) → (N-1-j, N-1-i)

    stack = np.stack([enhanced, vertical, horizontal, main_diag, anti_diag])
    return stack > ski.filters.threshold_otsu(stack)


def image_augmentation_1(image: np.ndarray) -> np.ndarray:
    """Augment a pre-processed image with 4 mirror flips.

    Unlike image_manipulation_take_2, this operates on an already-processed
    image rather than a raw RGB image — use after preprocessing.

    Args:
        image: 2-D grayscale or binary array of shape (H, W).
    Returns:
        Array of shape (5, H, W): original + 4 flips.
    """
    vertical   = np.flipud(image)
    horizontal = np.fliplr(image)
    main_diag  = image.T
    anti_diag  = image[::-1, ::-1].T
    return np.stack([image, vertical, horizontal, main_diag, anti_diag])
