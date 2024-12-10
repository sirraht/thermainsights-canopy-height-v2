import argparse
import os
import json
import rasterio
import numpy as np
import pandas as pd
from rasterio.windows import Window
from PIL import Image, ImageOps
from glob import glob

class Preprocess:
    def __init__(self, input_dir, output_dir, tile_size=256, border_pixels=10):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.border_pixels = border_pixels

    def save_image(self, image_array, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        image.save(output_path)

    def pad_or_crop_to_size(self, img_data, target_size):
        img = Image.fromarray(img_data.astype('uint8'), 'RGB')

        if img.size[0] < target_size[0] or img.size[1] < target_size[1]:
            padding = (0, 0, target_size[0] - img.size[0], target_size[1] - img.size[1])
            img = ImageOps.expand(img, padding, fill=(0, 0, 0))

        img = img.crop((0, 0, target_size[0], target_size[1]))
        return np.array(img)

    def split_geotiff(self, geotiff_path, output_subdir):
        tile_size = self.tile_size
        border_pixels = self.border_pixels

        if not (0 <= border_pixels < 50):
            raise ValueError("border_pixels must be between 0 and 50")

        extra_context = int(tile_size * (border_pixels / 100.0))

        with rasterio.open(geotiff_path) as src:
            width = src.width
            height = src.height

            # Extract and save geospatial metadata
            original_transform = src.transform
            crs = src.crs

            metadata = {
                "filename": os.path.basename(geotiff_path),
                "transform": original_transform.to_gdal(),
                "crs": crs.to_string(),
                "width": width,
                "height": height,
                "tile_size": tile_size,
                "border_pixels": border_pixels,
                "extra_context": extra_context
            }
            metadata_filename = os.path.join(output_subdir, 'image_metadata.json')
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f)

            tile_records = []
            tile_index = 0

            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    window = Window(
                        max(0, x - extra_context),
                        max(0, y - extra_context),
                        min(tile_size + 2 * extra_context, width - x + extra_context),
                        min(tile_size + 2 * extra_context, height - y + extra_context)
                    )

                    img_data = src.read([1, 2, 3], window=window)

                    if img_data.size == 0:
                        continue

                    img_data = np.moveaxis(img_data, 0, -1)
                    out_size = (tile_size + 2 * extra_context, tile_size + 2 * extra_context)
                    img_data = self.pad_or_crop_to_size(img_data, out_size)

                    x_index = x // tile_size
                    y_index = y // tile_size
                    input_tile_filename = f'tile_{y_index}_{x_index}_context.png'
                    pred_tile_filename = f'tile_{y_index}_{x_index}_context_pred.tif'

                    local_output_path = os.path.join(output_subdir, input_tile_filename)
                    self.save_image(img_data, local_output_path)

                    record = {
                        'x_index': x_index,
                        'y_index': y_index,
                        'input_filename': input_tile_filename,
                        'output_filename': pred_tile_filename,  # Predicted tile filename expected after inference
                        'imsize': tile_size + 2 * extra_context,
                        'bord_x': extra_context,
                        'bord_y': extra_context
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
    parser.add_argument('--tile_size', type=int, default=256, help='Size of each tile in pixels (default: 256).')
    parser.add_argument('--border_pixels', type=int, default=10, help='Percentage of border pixels for extra context (default: 10).')
    args = parser.parse_args()

    preprocessor = Preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        border_pixels=args.border_pixels
    )
    preprocessor.run()

if __name__ == '__main__':
    main()
