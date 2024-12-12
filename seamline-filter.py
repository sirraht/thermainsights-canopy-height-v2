import argparse
import os
import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from guided_filter import guided_filter
    GUIDED_FILTER_AVAILABLE = True
except ImportError:
    GUIDED_FILTER_AVAILABLE = False

def apply_gaussian_blur(img, sigma=1.0):
    return gaussian_filter(img, sigma=sigma)

def apply_bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Applies a bilateral filter to the image.
    Parameters:
      d: Diameter of each pixel neighborhood used during filtering.
      sigmaColor: Filter sigma in the color space.
      sigmaSpace: Filter sigma in the coordinate space.
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV not found. Install opencv-python to use bilateral filter.")
    # OpenCV expects uint8 images for bilateral by default.
    # Since we have float32, we can scale and convert.
    img_min, img_max = img.min(), img.max()
    # Avoid division by zero if img_max == img_min
    scale = 255.0 / (img_max - img_min) if img_max > img_min else 1.0
    img_uint8 = ((img - img_min) * scale).astype(np.uint8)
    filtered_uint8 = cv2.bilateralFilter(img_uint8, d, sigmaColor, sigmaSpace)
    filtered = filtered_uint8.astype(np.float32) / scale + img_min
    return filtered

def apply_guided_filter(img, radius=8, eps=0.01):
    """
    Applies a guided filter to the image using itself as the guidance image.
    Parameters:
      radius: The radius of the guided filter.
      eps: Regularization parameter.
    """
    if not GUIDED_FILTER_AVAILABLE:
        raise RuntimeError("guided-filter not found. Install guided-filter to use guided filter.")
    # guided_filter expects guidance and input as float arrays [0,1].
    img_min, img_max = img.min(), img.max()
    # Normalize to [0,1]
    denom = (img_max - img_min) if (img_max > img_min) else 1.0
    img_norm = (img - img_min) / denom
    # Apply guided filter
    filtered_norm = guided_filter(img_norm, img_norm, radius, eps)
    # Scale back
    filtered = filtered_norm * denom + img_min
    return filtered

def main():
    parser = argparse.ArgumentParser(description='Apply post-processing filters to reduce seamline artifacts.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input GeoTIFF.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the filtered GeoTIFF.')
    parser.add_argument('--gaussian', action='store_true', help='Apply Gaussian blur.')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma for Gaussian blur.')
    parser.add_argument('--bilateral', action='store_true', help='Apply Bilateral filter.')
    parser.add_argument('--d', type=int, default=9, help='Diameter of pixel neighborhood for bilateral filter.')
    parser.add_argument('--sigmaColor', type=float, default=75.0, help='Filter sigma in the color space for bilateral filter.')
    parser.add_argument('--sigmaSpace', type=float, default=75.0, help='Filter sigma in the coordinate space for bilateral filter.')
    parser.add_argument('--guided', action='store_true', help='Apply Guided filter.')
    parser.add_argument('--radius', type=int, default=8, help='Radius for the guided filter.')
    parser.add_argument('--eps', type=float, default=0.01, help='Regularization parameter for guided filter.')
    parser.add_argument('--all_filters', action='store_true', help='Apply all three filters in sequence: Gaussian -> Bilateral -> Guided.')
    args = parser.parse_args()

    # Read input
    with rasterio.open(args.input) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile

    # Apply filters based on arguments
    if args.all_filters:
        # Apply Gaussian -> Bilateral -> Guided in sequence
        print("Applying Gaussian blur...")
        img = apply_gaussian_blur(img, sigma=args.sigma)
        print("Applying Bilateral filter...")
        img = apply_bilateral_filter(img, d=args.d, sigmaColor=args.sigmaColor, sigmaSpace=args.sigmaSpace)
        print("Applying Guided filter...")
        img = apply_guided_filter(img, radius=args.radius, eps=args.eps)
    else:
        if args.gaussian:
            print("Applying Gaussian blur...")
            img = apply_gaussian_blur(img, sigma=args.sigma)
        if args.bilateral:
            print("Applying Bilateral filter...")
            img = apply_bilateral_filter(img, d=args.d, sigmaColor=args.sigmaColor, sigmaSpace=args.sigmaSpace)
        if args.guided:
            print("Applying Guided filter...")
            img = apply_guided_filter(img, radius=args.radius, eps=args.eps)

    # Save the filtered image
    profile.update(dtype='float32')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with rasterio.open(args.output, 'w', **profile) as dst:
        dst.write(img.astype(np.float32), 1)

    print(f"Filtered image saved to {args.output}")

if __name__ == '__main__':
    main()
