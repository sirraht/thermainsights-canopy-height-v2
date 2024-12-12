import argparse
import os
import json
import rasterio
import pandas as pd
import numpy as np
from tqdm import tqdm

class ReassembleToImage:
    def __init__(self, tiles_dir, original_tiles_dir, output_dir, metadata_json_path=None):
        """
        tiles_dir: Directory containing the predicted tiles.
        original_tiles_dir: Directory containing the original tiles and metadata JSON 
                            (only used if metadata_json_path is not provided).
        output_dir: Directory to save the reassembled GeoTIFF.
        metadata_json_path: Direct path to image_metadata.json.
        """
        self.tiles_dir = tiles_dir
        self.original_tiles_dir = original_tiles_dir
        self.output_dir = output_dir
        self.metadata_json_path = metadata_json_path

    def reassemble_tiles_in_folder(self, inferred_tiles_dir, metadata_json, output_subdir):
        # Load metadata
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)

        original_transform = rasterio.Affine.from_gdal(*metadata['transform'])
        crs = rasterio.crs.CRS.from_string(metadata['crs'])
        original_width = metadata['width']
        original_height = metadata['height']
        tile_size = metadata['tile_size']

        # Load tile metadata CSV
        metadata_csv = os.path.join(os.path.dirname(metadata_json), 'tile_metadata.csv')
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"{metadata_csv} not found.")

        metadata_df = pd.read_csv(metadata_csv)

        # Create an empty array for the reassembled image (float32)
        reassembled_image = np.zeros((original_height, original_width), dtype=np.float32)

        # Iterate over each tile record and place it into the final mosaic
        for _, row in tqdm(metadata_df.iterrows(), desc=f'Reassembling tiles in {inferred_tiles_dir}', total=len(metadata_df)):
            tile_path = os.path.join(inferred_tiles_dir, row['output_filename'])
            if not os.path.exists(tile_path):
                print(f"Tile {tile_path} not found, skipping.")
                continue

            # Read the predicted tile using rasterio to correctly handle float data
            with rasterio.open(tile_path) as src_tile:
                tile_array = src_tile.read(1).astype(np.float32)
            
            x_index = row['x_index']
            y_index = row['y_index']

            # Calculate the position where this tile should be placed in the mosaic
            x_pos = x_index * tile_size
            y_pos = y_index * tile_size

            # Determine the region to place the tile
            tile_h, tile_w = tile_array.shape
            y_end = min(y_pos + tile_h, original_height)
            x_end = min(x_pos + tile_w, original_width)

            # Crop the tile if it extends beyond image boundary
            tile_array = tile_array[0:(y_end - y_pos), 0:(x_end - x_pos)]

            reassembled_image[y_pos:y_end, x_pos:x_end] = tile_array

        os.makedirs(output_subdir, exist_ok=True)
        output_filepath = os.path.join(output_subdir, "reassembled_canopy_height.tif")

        # Write out the reassembled image as a float32 GeoTIFF
        with rasterio.open(
            output_filepath,
            'w',
            driver='GTiff',
            height=reassembled_image.shape[0],
            width=reassembled_image.shape[1],
            count=1,
            dtype='float32',
            crs=crs,
            transform=original_transform
        ) as dst:
            dst.write(reassembled_image, 1)

        print(f"Reassembled image saved to {output_filepath}")

    def run(self):
        if self.metadata_json_path is not None:
            # If metadata_json_path is given, assume tiles_dir points directly to inferred tiles
            folder_name = os.path.basename(os.path.normpath(self.tiles_dir))
            output_subdir = os.path.join(self.output_dir, folder_name)
            os.makedirs(output_subdir, exist_ok=True)
            self.reassemble_tiles_in_folder(self.tiles_dir, self.metadata_json_path, output_subdir)
        else:
            # Otherwise, look for inferred tiles directories inside tiles_dir
            inferred_tiles_folders = [
                os.path.join(self.tiles_dir, d) for d in os.listdir(self.tiles_dir) if os.path.isdir(os.path.join(self.tiles_dir, d))
            ]
            if not inferred_tiles_folders:
                print(f"No inferred tiles directories found in {self.tiles_dir}")
                return

            for inferred_tiles_dir in inferred_tiles_folders:
                folder_name = os.path.basename(inferred_tiles_dir)
                metadata_json = os.path.join(self.original_tiles_dir, folder_name, 'image_metadata.json')
                if not os.path.exists(metadata_json):
                    print(f"Metadata JSON not found in {metadata_json}, skipping")
                    continue

                output_subdir = os.path.join(self.output_dir, folder_name)
                os.makedirs(output_subdir, exist_ok=True)
                self.reassemble_tiles_in_folder(inferred_tiles_dir, metadata_json, output_subdir)

def main():
    parser = argparse.ArgumentParser(description='Reassemble inferred tiles into a single georeferenced image.')
    parser.add_argument('--tiles_dir', type=str, required=True, help='Directory containing the predicted canopy height tiles or a directory of subfolders with predicted tiles.')
    parser.add_argument('--original_tiles_dir', type=str, required=False, help='Directory containing the original tiles and metadata JSON if metadata_json is not provided.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the reassembled GeoTIFF.')
    parser.add_argument('--metadata_json', type=str, required=False, help='Path to the image_metadata.json file. If provided, tiles_dir should point directly to the directory containing predicted tiles.')
    args = parser.parse_args()

    reassembler = ReassembleToImage(
        tiles_dir=args.tiles_dir,
        original_tiles_dir=args.original_tiles_dir if args.original_tiles_dir else '',
        output_dir=args.output_dir,
        metadata_json_path=args.metadata_json
    )
    reassembler.run()

if __name__ == '__main__':
    main()
