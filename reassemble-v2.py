import argparse
import os
import json
import rasterio
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

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

    def create_weight_mask(self, tile_size, tile_overlap):
        """
        Create a weight mask for a tile given its size and overlap.
        The mask should be highest in the center and taper down toward edges.
        This ensures a smooth blend in overlapping regions.
        
        For simplicity, we'll use a cosine taper in the overlap region.
        If tile_overlap == 0, the mask is simply all ones.
        """

        if tile_overlap <= 0:
            # No overlap, just return a mask of ones
            return np.ones((tile_size, tile_size), dtype=np.float32)

        # Create 1D weight vectors for horizontal and vertical directions
        # Start from 0 to tile_size-1
        coords = np.arange(tile_size, dtype=np.float32)

        # Define a function that returns a weight between 0 and 1 depending on distance to edge
        # We'll define "effective overlap" regions near each edge: [0, tile_overlap) and [tile_size - tile_overlap, tile_size)
        # Use a cosine taper in these regions.
        def taper_func(x, length):
            # x in [0, length), return from 0 to 1 with a half-cosine shape
            # weight = 0.5*(1 - cos(pi*x/length))
            return 0.5 * (1 - np.cos((math.pi * x) / length))

        # Build horizontal weight:
        # For the left overlap region (0 to tile_overlap), weight ramps up from 0 to 1
        # For the right overlap region (tile_size-tile_overlap to tile_size), weight ramps down from 1 to 0
        # In the center region, weight = 1
        horiz_weights = np.ones(tile_size, dtype=np.float32)
        # Left overlap region
        left_mask = coords < tile_overlap
        horiz_weights[left_mask] = taper_func(coords[left_mask], tile_overlap)
        # Right overlap region
        right_mask = coords >= (tile_size - tile_overlap)
        horiz_weights[right_mask] = taper_func(tile_size - 1 - coords[right_mask], tile_overlap)

        # Vertical weight is the same logic
        vert_weights = np.ones(tile_size, dtype=np.float32)
        vert_weights[left_mask] = taper_func(coords[left_mask], tile_overlap)
        vert_weights[right_mask] = taper_func(tile_size - 1 - coords[right_mask], tile_overlap)

        # Combine horizontal and vertical into a 2D mask
        # outer product of horiz_weights and vert_weights
        weight_mask = np.outer(vert_weights, horiz_weights)

        # Normalize to max 1 just to be safe (should already be)
        max_val = weight_mask.max()
        if max_val > 0:
            weight_mask = weight_mask / max_val

        return weight_mask.astype(np.float32)

    def reassemble_tiles_in_folder(self, inferred_tiles_dir, metadata_json, output_subdir):
        # Load metadata
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)

        original_transform = rasterio.Affine.from_gdal(*metadata['transform'])
        crs = rasterio.crs.CRS.from_string(metadata['crs'])
        original_width = metadata['width']
        original_height = metadata['height']
        tile_size = metadata['tile_size']
        tile_overlap = metadata.get('tile_overlap', 0)  # Default to 0 if not in metadata

        # Load tile metadata CSV
        metadata_csv = os.path.join(os.path.dirname(metadata_json), 'tile_metadata.csv')
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"{metadata_csv} not found.")

        metadata_df = pd.read_csv(metadata_csv)

        # Instead of a direct assignment, we'll do weighted blending
        # Create sum and weight arrays
        reassembled_sum = np.zeros((original_height, original_width), dtype=np.float32)
        reassembled_weight_sum = np.zeros((original_height, original_width), dtype=np.float32)

        # Precompute the weight mask for a full tile
        tile_weight_mask = self.create_weight_mask(tile_size, tile_overlap)

        # Iterate over each tile
        for _, row in tqdm(metadata_df.iterrows(), desc=f'Reassembling tiles in {inferred_tiles_dir}', total=len(metadata_df)):
            tile_path = os.path.join(inferred_tiles_dir, row['output_filename'])
            if not os.path.exists(tile_path):
                print(f"Tile {tile_path} not found, skipping.")
                continue

            # Read the predicted tile
            with rasterio.open(tile_path) as src_tile:
                tile_array = src_tile.read(1).astype(np.float32)
            
            x_index = row['x_index']
            y_index = row['y_index']

            # Calculate position of this tile in the mosaic
            # Note: The stride used in preprocessing was (tile_size - tile_overlap)
            # Recover stride from these parameters if needed
            stride = tile_size - tile_overlap
            x_pos = x_index * stride
            y_pos = y_index * stride

            tile_h, tile_w = tile_array.shape
            y_end = min(y_pos + tile_h, original_height)
            x_end = min(x_pos + tile_w, original_width)

            # Crop if needed (in case last tile is smaller)
            tile_array = tile_array[:(y_end - y_pos), :(x_end - x_pos)]
            tile_local_mask = tile_weight_mask[:(y_end - y_pos), :(x_end - x_pos)]

            # Add weighted tile to sum and weight arrays
            reassembled_sum[y_pos:y_end, x_pos:x_end] += tile_array * tile_local_mask
            reassembled_weight_sum[y_pos:y_end, x_pos:x_end] += tile_local_mask

        # Compute final blended image
        # Avoid division by zero
        zero_mask = (reassembled_weight_sum == 0)
        reassembled_weight_sum[zero_mask] = 1
        final_image = reassembled_sum / reassembled_weight_sum
        # Where weight was zero (no tiles), final_image will be meaningless, but that should not happen if coverage is correct.

        # Write out the final image
        os.makedirs(output_subdir, exist_ok=True)
        output_filepath = os.path.join(output_subdir, "reassembled_canopy_height.tif")

        with rasterio.open(
            output_filepath,
            'w',
            driver='GTiff',
            height=final_image.shape[0],
            width=final_image.shape[1],
            count=1,
            dtype='float32',
            crs=crs,
            transform=original_transform
        ) as dst:
            dst.write(final_image, 1)

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
                    print(f"Metadata JSON not found at {metadata_json}, skipping")
                    continue

                output_subdir = os.path.join(self.output_dir, folder_name)
                os.makedirs(output_subdir, exist_ok=True)
                self.reassemble_tiles_in_folder(inferred_tiles_dir, metadata_json, output_subdir)

def main():
    parser = argparse.ArgumentParser(description='Reassemble inferred tiles into a single georeferenced image with blending.')
    parser.add_argument('--tiles_dir', type=str, required=True, help='Directory containing predicted canopy height tiles or a directory of subfolders with predicted tiles.')
    parser.add_argument('--original_tiles_dir', type=str, required=False, help='Directory containing the original tiles and metadata JSON if metadata_json is not provided.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the reassembled (blended) GeoTIFF.')
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
