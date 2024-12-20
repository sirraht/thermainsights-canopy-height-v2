import argparse
import os
import json
import rasterio
import numpy as np
import pandas as pd
from rasterio.windows import Window
from PIL import Image
from glob import glob

class Preprocess:
    def __init__(self, input_dir, output_dir, user_tile_size):
        # Hardcode these values as requested
        self.patch_size = 256
        self.overlap = 64
        self.step = self.patch_size - self.overlap  # 192
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Determine final tile size based on user input
        # If user_tile_size is provided, we snap it to a multiple of self.step that is >= self.patch_size
        if user_tile_size is not None:
            # Ensure user_tile_size is at least patch_size to have room for overlaps
            user_tile_size = max(user_tile_size, self.patch_size)
            # Round user_tile_size down to nearest multiple of step that is >= patch_size
            final_tile_size = (user_tile_size // self.step) * self.step
            if final_tile_size < self.patch_size:
                final_tile_size = self.patch_size
        else:
            # Default tile size if none provided: 1024 is a common choice
            user_tile_size = 1024
            final_tile_size = (user_tile_size // self.step) * self.step
            if final_tile_size < self.patch_size:
                final_tile_size = self.patch_size

        self.tile_size = final_tile_size
        print(f"Using a tile size of {self.tile_size}x{self.tile_size}, step={self.step} for splitting.")

    def save_image(self, image_array, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        image.save(output_path)

    def split_geotiff(self, geotiff_path, output_subdir):
        tile_size = self.tile_size
        step = self.step

        with rasterio.open(geotiff_path) as src:
            width = src.width
            height = src.height
            original_transform = src.transform
            crs = src.crs

            # Compute cropped dimensions that are multiples of step and <= original size
            new_width = (width // step) * step
            new_height = (height // step) * step

            if new_width < self.patch_size:
                new_width = self.patch_size
            if new_height < self.patch_size:
                new_height = self.patch_size

            # If input is smaller than the chosen tile size, we just proceed with what we have
            if new_width < tile_size:
                tile_size = new_width
            if new_height < tile_size:
                tile_size = new_height

            # Log the cropping decision
            if (new_width < width) or (new_height < height):
                print(f"Cropping image {os.path.basename(geotiff_path)} from {width}x{height} "
                      f"to {new_width}x{new_height} so dimensions are multiples of {step}.")
                print("This ensures that when we run overlapping inference patches, "
                      "the entire image will be fully covered without gaps.")
            else:
                print(f"Image {os.path.basename(geotiff_path)} dimensions are already multiples of {step}, "
                      f"no cropping needed. Dimensions: {width}x{height}")

            # Read the entire image
            full_img = src.read([1, 2, 3])  # shape: (3, H, W)
            full_img = np.moveaxis(full_img, 0, -1)  # shape: (H, W, 3)

        # Crop the image in-memory
        cropped_img = full_img[:new_height, :new_width, :]

        # Update metadata to reflect the cropped dimensions
        metadata = {
            "filename": os.path.basename(geotiff_path),
            "transform": original_transform.to_gdal(),
            "crs": crs.to_string(),
            "width": new_width,
            "height": new_height,
            "tile_size": tile_size
        }
        metadata_filename = os.path.join(output_subdir, 'image_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f)

        tile_records = []
        tile_index = 0

        # Now split the cropped image into possibly larger tiles (tile_size x tile_size)
        # These tiles will later be broken down into 256x256 patches with overlap
        for y in range(0, new_height, tile_size):
            for x in range(0, new_width, tile_size):
                # Ensure we don't go out of bounds if the final segment is smaller than tile_size
                tile_w = min(tile_size, new_width - x)
                tile_h = min(tile_size, new_height - y)
                img_data = cropped_img[y:y+tile_h, x:x+tile_w, :]

                y_index = y // tile_size
                x_index = x // tile_size
                input_tile_filename = f'tile_{y_index}_{x_index}_context.png'
                pred_tile_filename = f'tile_{y_index}_{x_index}_context_pred.tif'

                local_output_path = os.path.join(output_subdir, input_tile_filename)
                self.save_image(img_data, local_output_path)

                record = {
                    'x_index': x_index,
                    'y_index': y_index,
                    'input_filename': input_tile_filename,
                    'output_filename': pred_tile_filename,
                    'imsize': tile_size,
                    'bord_x': 0,
                    'bord_y': 0
                }
                tile_records.append(record)
                tile_index += 1

        if tile_records:
            df = pd.DataFrame(tile_records)
            csv_output_path = os.path.join(output_subdir, 'tile_metadata.csv')
            df.to_csv(csv_output_path, index=False)
        else:
            print(f"No tiles were processed for {geotiff_path}. No metadata CSV created.")

    def run(self):
        geotiff_files = glob(os.path.join(self.input_dir, '*.tif'))
        if not geotiff_files:
            print(f"No GeoTIFF files found in {self.input_dir}")
            return

        for geotiff_path in geotiff_files:
            filename = os.path.basename(geotiff_path)
            file_base, _ = os.path.splitext(filename)
            output_subdir = os.path.join(self.output_dir, file_base)
            os.makedirs(output_subdir, exist_ok=True)

            print(f"Processing file {geotiff_path}")
            self.split_geotiff(geotiff_path, output_subdir)

def main():
    parser = argparse.ArgumentParser(description="Preprocess aerial images for canopy height inference.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input GeoTIFF images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed tiles and CSV.')
    parser.add_argument('--user_tile_size', type=int, default=None,
                        help='Optional desired tile dimension (e.g. 1000 means produce ~1000x1000 tiles). '
                             'Will be snapped to a multiple of step (256-overlap=192) under the hood.')
    args = parser.parse_args()

    preprocessor = Preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        user_tile_size=args.user_tile_size
    )
    preprocessor.run()

if __name__ == '__main__':
    main()
