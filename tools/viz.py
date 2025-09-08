import cv2
import matplotlib.pyplot as plt

### Some of the functions are copied from hloc
def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image

def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True, figsize=4.5):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*figsize, figsize]
    fig, axs = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        axs = [axs]
    for i, (img, ax) in enumerate(zip(imgs, axs)):
        ax.imshow(img, cmap=plt.get_cmap(cmaps[i]))
        ax.set_axis_off()
        if titles:
            ax.set_title(titles[i])
    fig.tight_layout(pad=pad)

def plot_sequence(images, figsize=(15, 10), dpi=100, pad=.5, show = True, label = None):
    """Plot a set of image sequences where horizontal images are from the same sequence.
    Args:
        images: a list of list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        
    """
    n = len(images) # num of sequences
    l = len(images[0]) # num of images in each sequence
    
    fig, axs = plt.subplots(n, l, figsize=figsize, dpi=dpi)
    
    if n == 1:
        axs = [axs]
    for i, (image, ax) in enumerate(zip(images, axs)):
        for j, img in enumerate(image):
            ax[j].imshow(img)
            # ax[j].set_title(f'Image {j}')
        # ax.imshow(image)
            ax[j].set_axis_off()
    fig.tight_layout(pad=pad)
    
    if label is not None:
        fig.suptitle(f'This is a {label} pair.', fontsize=16)
    if show:
        plt.show()

def plot_retreivals(queries, retreivals, positives, figsize=(15, 10), dpi=100, pad=.5):
    """Plot a set of image sequences where horizontal images are from the same sequence.
    Args:
        queries: a list of list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        retreivals: a list of list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        positives: a list of strings, as titles of postives ground truth.
    """
    n = len(queries) # num of queries
    l = len(retreivals[0]) # top l retreivals
    
    fig, axs = plt.subplots(n, l+1, figsize=figsize, dpi=dpi)
    
    if n == 1:
        axs = [axs]
    for i, (retreival, ax) in enumerate(zip(retreivals, axs)):
        ax[0].imshow(queries[i])
        ax[0].set_title('Query')
        ax[0].set_axis_off()
        for j, (img, pos)  in enumerate(zip(retreival,positives)):
            ax[j+1].imshow(img)
            ax[j+1].set_title(f'Image {j}: {pos} Pair')
            ax[j+1].set_axis_off()
    fig.tight_layout(pad=pad)