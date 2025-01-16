# Canopy Height Model Processing Pipeline

A Python-based pipeline for processing and converting canopy height model (CHM) data into Cloud-Optimized GeoTIFFs (COGs) for efficient streaming via titiler on AWS Lambda.

Forked from [Meta High Res Canopy Height](https://github.com/facebookresearch/HighResCanopyHeight).

## Overview

This project provides tools to:
1. Preprocess aerial imagery into tiles
2. Run canopy height inference using a Meta and WRI vision transformer model
3. Reassemble predicted tiles into complete images
4. Convert GeoTIFFs to Cloud-Optimized GeoTIFFs (COGs)

## Key Components

### 1. Preprocessing
The preprocessing module (`preprocess-v2.py`) handles:
- Splitting large aerial images into overlapping tiles
- Managing tile metadata and coordinates
- Preparing data for model inference

### 2. Inference
The aerial inference module (`aerial-inference-v2.py`) includes:
- Vision Transformer backbone for feature extraction
- DPT head for height prediction
- Weighted patch blending for seamless predictions

### 3. Reassembly
The reassembly module (`reassemble-v2.py`) provides:
- Tile reassembly with smooth blending
- Preservation of geospatial metadata
- Output of complete GeoTIFF images

### 4. COG Conversion
The COG conversion module (`geotif-to-cog.py`) handles:
- Conversion of GeoTIFFs to Cloud-Optimized format
- Proper byte scaling (0-255)
- Removal of nodata values
- Web-optimized tiling

## Installation

```bash
# Clone the repository
git clone [repository-url]
```
Follow instructions on the Meta repo for conda virtual environment creation and dependency installation.

Access the pretrained model for aerial imagery inference via AWS S3 as noted in the Meta repo.  

In the saved_checkpoints directory that you've downloaded, you'll find:
* compressed_SSLhuge_aerial.pth (749M): The pretrained model you'll want to use for canopy height inference on aerial imagery, encoder trained on satellite images, decoder trained on aerial images.  
* aerial_normalization_quantiles_predictor.ckpt: A model provided by Meta to predict the 95th and 5th percentiles of the corresponding images to automate color balancing.

## Usage

### 1. Preprocess Images
```bash
python preprocess-v2.py \
  --input_dir /path/to/input/images \
  --output_dir /path/to/output/tiles \
  --tile_size 256 \
  --overlap 64
```

### 2. Run Inference
```bash
python aerial-inference-v2.py \
  --checkpoint saved_checkpoints/model.pth \
  --preprocessed_dir /path/to/preprocessed/data \
  --image_dir /path/to/image/tiles
```

### 3. Reassemble Tiles
```bash
python reassemble-v2.py \
  --tiles_dir /path/to/predicted/tiles \
  --output_dir /path/to/output \
  --metadata_json /path/to/metadata.json
```

### 4. Convert to COG
```bash
python geotif-to-cog.py \
  --input input.tif \
  --output output.tif \
  --tile_size 256
```

## License

This code and model weights are licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project builds on and utilizes key technology from:
- [Meta High Res Canopy Height](https://github.com/facebookresearch/HighResCanopyHeight)
