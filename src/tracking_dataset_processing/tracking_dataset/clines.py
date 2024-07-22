# Third-party libraries
import cv2
import numpy as np
import skimage
from scipy import interpolate


def extract_only_largest_contour(mask):
    """Extract only the largest contour from the mask.
    
    Args:
        mask: np.array, mask of the object.
    
    Returns:
        mask: np.array, mask of the object with only the largest contour.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(mask)
    
    if len(contours) > 0:
        contour = contours[np.argmax([cv2.contourArea(c) for c in contours])]
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
    
    return mask

def extract_contour_points_and_normals(mask, num_points_on_contour=200):
    """Extract contour points and normals from the mask.
        1. Get the largest contour.
        2. Interpolate the contour to compute normals.
    
    Args:
        mask: np.array, mask of the object.
        num_points_on_contour: int, number of points on the contour.
    
    Returns:
        points: np.array, [nx2] points on the contour.
        normals: np.array, [nx2] normals to the contour.
    """
    # Get the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour = contours[np.argmax([cv2.contourArea(c) for c in contours])][:, 0, :]

    # Interpolate the contour to compute normals
    tck, u = interpolate.splprep(contour.T, per=True)
    tii = np.linspace(
        0, contour.shape[0], num_points_on_contour, endpoint=False, dtype=int
    )
    points = contour[tii].astype(np.float32)

    tangents = np.asarray(interpolate.splev(u[tii], tck, der=1)).T
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = (np.asarray([[0, -1], [1, 0]]) @ tangents.T).T

    return points, normals

def _line(p, t, n=100):
    """
    Given a point (x,y) and tangent(dx,dy) return a set of points M*2
    """
    return np.asarray(
        skimage.draw.line(
            *np.round(p[::-1], decimals=0).astype(int),
            *np.round(p[::-1] + n * t[::-1], decimals=0).astype(int),
        )
    ).T

def extract_contour_lines(points, normals, line_size_half):
    """Extract contour lines from points and normals.
    
    Args:
        points: np.array, [nx2] points on the contour.
        normals: np.array, [nx2] normals to the contour.
        line_size_half: int, half size of the line.
    
    Returns:
        clines: np.array, [nx2*line_size_halfx2] contour lines.
    """
    # from float normals to lines in image
    outer_lines = [
        _line(p, t, n=line_size_half * 10)[1 : line_size_half + 1]
        for p, t in zip(points, normals)
    ]
    inner_lines = [
        _line(p, -t, n=line_size_half * 10)[:line_size_half][::-1]
        for p, t in zip(points, normals)
    ]
    try:
        outer_lines = np.stack(outer_lines)
        inner_lines = np.stack(inner_lines)
    except ValueError:
        print("not enough points for some of the line")
        exit(1)
    clines = np.concatenate((inner_lines, outer_lines), axis=1)

    return clines

def random_homography_from_points(points, scale=0.2):
    """
    Generate random homography from points by extracting bounding box and corrupted it
    by noise.
    
    Args:
        points: np.array, [nx2] points.
        scale: float, scale of the noise.
    
    Returns:
        H: np.array, [3x3] homography matrix.
    """
    xmin, xmax = np.min(points[:, 0]), np.max(points[:, 0])
    ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
    orig_bb = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    new_bb = (
        orig_bb
        + np.random.uniform(-1, 1, size=(4, 2))
        * np.array([xmax - xmin, ymax - ymin])
        * scale
    )
    H, _ = cv2.findHomography(orig_bb, new_bb)
    
    return H

def apply_homography(points, H):
    """
    Apply homography to points.
    """
    v = H @ np.append(points, np.ones((points.shape[0], 1)), axis=1).T
    
    return (v[:2, :] / v[2, :]).T

def apply_homography_to_points_with_normals(points, normals, H):
    """
    Apply homography to points and normals.
    """
    points_transformed = apply_homography(points, H)
    tmp = apply_homography(points + normals, H)
    normals_transformed = tmp - points_transformed
    normals_transformed /= np.linalg.norm(normals_transformed, axis=1, keepdims=True)
    
    return points_transformed, normals_transformed
