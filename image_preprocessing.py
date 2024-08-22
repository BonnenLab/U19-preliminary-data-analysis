import numpy as np
import math
from tqdm import tqdm

# Helper functions

def normalize_pixel_intensities(frames):
    frames = frames - np.array(frames).mean()
    frames = frames/np.abs(frames).max()/2
    frames = frames+.5
    frames = frames*255
    return frames.astype(np.uint8)

def crop_central_square(img):
    """
    Crops the central square region from an image.
    
    Args:
    img (numpy.ndarray): The input image as a 2D or 3D numpy array.
    
    Returns:
    numpy.ndarray: The cropped central square of the image.
    """
    # Calculate the smallest dimension to define the square's size
    width = min(img.shape[:2])
    # Determine the top-left coordinates to start cropping
    x = int(img.shape[0] / 2 - width / 2)
    y = int(img.shape[1] / 2 - width / 2)
    # Crop the square out of the image
    crop_img = img[x:x + width, y:y + width]
    return crop_img

def coswin(n, inner_radius, outer_radius):
    """
    Generates a cosine window mask in a square matrix.

    Args:
    n (tuple): Dimensions (height, width) of the output matrix.
    inner_radius (float): Radius where the cosine taper begins.
    outer_radius (float): Radius where the cosine taper ends and beyond which all values are zero.

    Returns:
    numpy.ndarray: Matrix with a central circular cosine window and tapering edges.
    """
    # Calculate radial increment in radians for cosine function application
    r2rad = math.pi / 2 * (1 / (outer_radius - inner_radius))
    
    nx,ny = n
    x = nx*np.linspace(-.5, .5, nx)
    y = ny*np.linspace(-.5, .5, ny)
    xv, yv = np.meshgrid(x, y)
    h = np.sqrt(xv**2+yv**2)
    W = np.ones(xv.shape)
    
    
    # set cosine ring
    idx = (h>inner_radius) & (h<=outer_radius)
    vals = np.cos((h-inner_radius)*r2rad)
    W[idx] = vals[idx]
    
    # set outer part to zero
    idx2 = h>outer_radius;
    W[idx2] = 0;
    return W.T # Transpose to match input dimensions

def crop_and_mask_video(frames):
    """
    Applies a cosine window mask to each frame in a list of frames and then crops to the central square.

    Args:
    frames (list of numpy.ndarray): List of image frames to process.

    Returns:
    numpy.ndarray: An array of processed frames, each masked and cropped.
    """
    masked_and_cropped = []

    print('Processing frames...\n')
    for ii in tqdm(range(frames.shape[0]-1)):
        img = frames[ii]

        # Create and apply a cosine window mask
        cwin = coswin(img.shape, 12, 14)
        masked = cwin * (img-255/2) + 255/2

        # Crop the central square of the masked image
        cropped = crop_central_square(masked)
        masked_and_cropped.append(cropped)

    # Stack all processed frames into a single numpy array
    frames_processed = np.stack(masked_and_cropped).astype(np.uint16)
    return frames_processed

def crop_video(frames):
    """
    Crops to the central square.

    Args:
    frames (list of numpy.ndarray): List of image frames to process.

    Returns:
    numpy.ndarray: An array of processed frames, each masked and cropped.
    """
    all = []

    print('Processing frames...\n')
    for ii in tqdm(range(frames.shape[0]-1)):
        img = frames[ii]

        # Crop the central square of the masked image
        cropped = crop_central_square(img)
        all.append(cropped)

    # Stack all processed frames into a single numpy array
    frames_processed = np.stack(all)
    return frames_processed