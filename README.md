# Canopy Height Model Processing Pipeline

A Python-based pipeline for processing and converting canopy height model (CHM) data into Cloud-Optimized GeoTIFFs (COGs) for efficient streaming via titiler on AWS Lambda.

## Overview

This project provides tools to:
1. Preprocess aerial imagery into tiles
2. Run canopy height inference using a vision transformer model
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

# Install dependencies
pip install -r requirements.txt
```

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

## Model Architecture

The system uses a Vision Transformer-based architecture with:
- SSL pre-trained backbone
- DPT-style decoder head
- Adaptive patch processing
- Weighted feature fusion

## Requirements

- Python 3.8+
- PyTorch
- rasterio
- rio-cogeo
- numpy
- pandas
- PIL

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## Acknowledgments

This project builds on several key technologies:
- [Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929)
- [Dense Prediction Transformers (DPT)](https://arxiv.org/abs/2103.13413)
- [Cloud-Optimized GeoTIFF specification](https://www.cogeo.org/)
- [TiTiler](https://github.com/developmentseed/titiler) for cloud-native tile serving