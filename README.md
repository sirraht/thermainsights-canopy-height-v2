Canopy Height Model Processing Pipeline
A Python-based pipeline for processing and converting canopy height model (CHM) data into Cloud-Optimized GeoTIFFs (COGs) for efficient streaming via titiler on AWS Lambda.
Overview
This project provides tools to:
Preprocess aerial imagery into tiles
Run canopy height inference using a vision transformer model
Reassemble predicted tiles into complete images
Convert GeoTIFFs to Cloud-Optimized GeoTIFFs (COGs)
Key Components
1. Preprocessing
The preprocessing module (preprocess-v2.py) handles:
Splitting large aerial images into overlapping tiles
Managing tile metadata and coordinates
Preparing data for model inference
2. Inference
The aerial inference module (aerial-inference-v2.py) includes:
Vision Transformer backbone for feature extraction
DPT head for height prediction
Weighted patch blending for seamless predictions
3. Reassembly
The reassembly module (reassemble-v2.py) provides:
Tile reassembly with smooth blending
Preservation of geospatial metadata
Output of complete GeoTIFF images
4. COG Conversion
The COG conversion module (geotif-to-cog.py) handles:
Conversion of GeoTIFFs to Cloud-Optimized format
Proper byte scaling (0-255)
Removal of nodata values
Web-optimized tiling
Installation
Usage
1. Preprocess Images
2. Run Inference
3. Reassemble Tiles
4. Convert to COG
Model Architecture
The system uses a Vision Transformer-based architecture with:
SSL pre-trained backbone
DPT-style decoder head
Adaptive patch processing
Weighted feature fusion
Requirements
Python 3.8+
PyTorch
rasterio
rio-cogeo
numpy
pandas
PIL
License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
Contributing
Contributions are welcome! Please read our contributing guidelines before submitting pull requests.
Acknowledgments
This project builds on several key technologies:
Vision Transformers (ViT)
Dense Prediction Transformers (DPT)
Cloud-Optimized GeoTIFF specification
TiTiler for cloud-native tile serving