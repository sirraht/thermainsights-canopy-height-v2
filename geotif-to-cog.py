import argparse
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles


def scale_to_byte(data, user_min=None, user_max=None):
    """
    Linearly scales an array to the 0-255 range (UInt8) using either provided user_min/user_max
    or the data's own min/max. Values between 0 and 1 are set to 0 before any scaling.
    """
    # Replace NaN values with 0 first
    data = np.nan_to_num(data, nan=0.0)
    
    # Set values between 0 and 1 to 0 BEFORE scaling
    data = np.where((data > 0) & (data < 1), 0, data)
    
    # Determine scaling bounds from valid data
    min_val = user_min if user_min is not None else data.min()
    max_val = user_max if user_max is not None else data.max()

    # Avoid division by zero if constant band
    if max_val - min_val == 0:
        return np.zeros(data.shape, dtype=np.uint8)

    # Clip values to the specified range before scaling
    clipped = np.clip(data, min_val, max_val)

    # Perform linear scaling
    scaled = ((clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return scaled


def convert_to_cog(input_file, output_file, tile_size=256, scale_min=None, scale_max=None):
    target_crs = "EPSG:3857"

    with rasterio.open(input_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        temp_kwargs = src.meta.copy()
        temp_kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": src.dtypes[0],
            "nodata": None  # Remove nodata value for intermediate step
        })

        with MemoryFile() as memfile:
            # Reproject into memory with original data type
            with memfile.open(**temp_kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                        dst_nodata=None  # Don't set nodata in reprojection
                    )

            # Open the reprojected dataset and prepare for byte scaling
            with memfile.open() as reprojected_ds:
                byte_kwargs = reprojected_ds.meta.copy()
                byte_kwargs.update({
                    "dtype": 'uint8',
                    "nodata": None  # Remove nodata value for final output
                })

                with MemoryFile() as byte_memfile:
                    with byte_memfile.open(**byte_kwargs) as byte_dst:
                        for i in range(1, reprojected_ds.count + 1):
                            band_data = reprojected_ds.read(i)  # Read without masking
                            scaled_data = scale_to_byte(
                                band_data,
                                user_min=scale_min,
                                user_max=scale_max
                            )
                            byte_dst.write(scaled_data, i)

                    cog_profile = cog_profiles.get("deflate")
                    cog_profile.update({
                        "blocksize": tile_size,
                        "nodata": None  # Ensure no nodata value in COG
                    })

                    cog_translate(
                        byte_memfile,
                        output_file,
                        cog_profile,
                        in_memory=True
                    )

    print(f"COG written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a GeoTIFF to a COG with byte scaling, eliminating NaNs and nodata."
    )
    parser.add_argument("input_file", type=str, help="Path to the input GeoTIFF file.")
    parser.add_argument("output_file", type=str, help="Path to the output COG file.")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size for the COG. Defaults to 256.")
    parser.add_argument("--scale_min", type=float, default=None, help="Minimum value for scaling. Defaults to data minimum.")
    parser.add_argument("--scale_max", type=float, default=None, help="Maximum value for scaling. Defaults to data maximum.")
    args = parser.parse_args()

    convert_to_cog(
        args.input_file,
        args.output_file,
        tile_size=args.tile_size,
        scale_min=args.scale_min,
        scale_max=args.scale_max
    )


if __name__ == "__main__":
    main()
