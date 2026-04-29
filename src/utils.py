import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt


def Image_Display(image, image2=None, image3=None, cmap='gray', title='an image', title2='an image', title3='an image'):
    """Display 1, 2, or 3 images side-by-side."""
    images = [img for img in [image, image2, image3] if img is not None]
    titles = [title, title2, title3][:len(images)]

    if len(images) == 1:
        plt.imshow(images[0], cmap=cmap)
        plt.title(titles[0])
    else:
        fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
        for ax, img, t in zip(axes, images, titles):
            ax.imshow(img, cmap=cmap)
            ax.set_title(t)
        plt.tight_layout()
    plt.show()
    plt.clf()


def var_memory_report():
    """Print a memory-usage table for all variables in the caller's global scope."""
    caller_globals = inspect.stack()[1][0].f_globals

    total = 0
    rows = []
    for name, obj in sorted(caller_globals.items()):
        if name.startswith('_'):
            continue
        size = obj.nbytes if isinstance(obj, np.ndarray) else sys.getsizeof(obj)
        total += size
        rows.append((name, type(obj).__name__, size))

    rows.sort(key=lambda x: -x[2])
    print(f"{'Variable':30s} {'Type':20s} {'Size (MB)':>10}")
    print("-" * 65)
    for name, typ, size in rows:
        print(f"{name:30s} {typ:20s} {size/1e6:>10.4f}")
    print("-" * 65)
    print(f"{'TOTAL':30s} {'':20s} {total/1e6:>10.4f}")


def visualize_kernel(image: np.ndarray, kernel_size: int, top_left: tuple) -> None:
    """Overlay a square kernel footprint on an image and display it.

    Args:
        image:       2-D grayscale array (H, W).
        kernel_size: Side length of the square kernel overlay in pixels.
        top_left:    (row, col) of the kernel's top-left corner.
    """
    row, col = top_left
    h, w = image.shape[:2]
    if row < 0 or col < 0 or row + kernel_size > h or col + kernel_size > w:
        raise ValueError(
            f"Kernel [{row}:{row+kernel_size}, {col}:{col+kernel_size}] "
            f"falls outside image bounds ({h}, {w})."
        )
    im = image.copy()
    im[row:row + kernel_size, col:col + kernel_size] = 0.5
    Image_Display(im, title=f'Kernel {kernel_size}x{kernel_size} at ({row}, {col})')
