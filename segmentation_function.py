import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed


def water_shed_segmentation(BW, threshold):
    """
    Performs watershed segmentation on a binary image using a threshold.

    Args:
        BW (numpy.ndarray): Binary image.
        threshold (float): Threshold value for segmentation.

    Returns:
        numpy.ndarray: Labeled segmented image.
    """
    D = -distance_transform_edt(BW)
    M = D < -threshold
    L = watershed(D, label(M), mask=BW)
    return L


def convex_judge(B):
    """
    Determines if a given boundary is convex.

    Args:
        B (numpy.ndarray): Nx2 array of boundary points.

    Returns:
        bool: True if the boundary is convex, False otherwise.
    """
    mask = np.zeros((512, 512), dtype=np.uint8)
    rr, cc = B[:, 1], B[:, 0]  # Rows and Columns
    cv2.fillPoly(mask, [np.array([cc, rr]).T], 1)
    props = regionprops(mask)
    return props[0].solidity > 0.9 if props else False


def process_image(image):
    """
    Processes the input image to extract convex contours.

    Args:
        image (numpy.ndarray): Input image (2D array).

    Returns:
        numpy.ndarray: Array of all convex contours (all_Bs_array).
    """
    all_Bs = []  # List to store all contours
    
    # Preprocessing masks
    M = cv2.resize(image, (512, 512))  # Resize to standard size
    S = (M > 50) & (M < 180)  # Non-overlapping particles
    D = (M > 180)  # Overlapping particles

    # Apply watershed segmentation
    D = water_shed_segmentation(D.astype(np.uint8) * 255, threshold=2.5)
    D = clear_border(D)
    D_labeled = label(D)

    S = water_shed_segmentation(S.astype(np.uint8) * 255, threshold=2.5)
    S = clear_border(S)
    S_labeled = label(S)

    # Extract contours
    Bs = []
    for j in range(1, np.max(S_labeled) + 1):
        single_ind = (S_labeled == j)  # Cluster of non-overlapping particles
        mask_or = single_ind | D
        mask_or = water_shed_segmentation(mask_or.astype(np.uint8) * 255, threshold=5)
        or_labeled = label(mask_or)

        for k in range(1, np.max(or_labeled) + 1):
            particle = (or_labeled == k)
            if np.sum(particle & single_ind) < 1:
                continue
            contours, _ = cv2.findContours(
                particle.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                B = contours[0][:, 0, :]  # Extract Nx2 points
                Bs.append(B)

    # Filter convex contours
    for B in Bs:
        if convex_judge(B):
            all_Bs.append(B)

    # Convert the list of all contours to a numpy array
    all_Bs_array = np.array(all_Bs, dtype=object)
    return all_Bs_array
