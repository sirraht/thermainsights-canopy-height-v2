import argparse
import os
import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter

def apply_gaussian_blur(input_tif, output_tif, sigma=1.0):
    # Read the input GeoTIFF
    with rasterio.open(input_tif) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile

    # Apply Gaussian blur
    # sigma controls the amount of blurring; a larger sigma = more blur.
    # You can also specify a kernel size indirectly via sigma.
    blurred = gaussian_filter(img, sigma=sigma)

    # Update profile as needed (usually unchanged)
    profile.update(dtype='float32')

    # Save the blurred image as a new GeoTIFF
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(blurred.astype(np.float32), 1)

def main():
    parser = argparse.ArgumentParser(description='Apply Gaussian blur to a GeoTIFF image.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input GeoTIFF (from reassemble-v2.py).')
    parser.add_argument('--output', type=str, required=True, help='Path to save the blurred GeoTIFF.')
    parser.add_argument('--sigma', type=float, default=1.0, help='Standard deviation for the Gaussian kernel (default: 1.0).')
    args = parser.parse_args()

    apply_gaussian_blur(args.input, args.output, sigma=args.sigma)

if __name__ == '__main__':
    main()
