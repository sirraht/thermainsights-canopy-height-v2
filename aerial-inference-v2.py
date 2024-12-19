import argparse
import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import rasterio
import torchvision
import pytorch_lightning as pl
from models.backbone import SSLVisionTransformer
from models.dpt_head import DPTHead
from models.regressor import RNet

class SSLAE(nn.Module):
    def __init__(self, pretrained=None, classify=True, n_bins=256, huge=False):
        super().__init__()
        if huge:
            self.backbone = SSLVisionTransformer(
                embed_dim=1280,
                num_heads=20,
                out_indices=(9, 16, 22, 29),
                depth=32,
                pretrained=pretrained
            )
            self.decode_head = DPTHead(
                classify=classify,
                in_channels=(1280, 1280, 1280, 1280),
                embed_dims=1280,
                post_process_channels=[160, 320, 640, 1280],
            )  
        else:
            self.backbone = SSLVisionTransformer(pretrained=pretrained)
            self.decode_head = DPTHead(classify=classify, n_bins=256)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x) 
        return x

class SSLModule(pl.LightningModule):
    def __init__(self, ssl_path="compressed_SSLbaseline.pth"):
        super().__init__()
    
        if 'huge' in ssl_path:
            self.chm_module_ = SSLAE(classify=True, huge=True).eval()
        else:
            self.chm_module_ = SSLAE(classify=True, huge=False).eval()
        
        if 'compressed' in ssl_path:   
            ckpt = torch.load(ssl_path, map_location='cpu')
            self.chm_module_ = torch.quantization.quantize_dynamic(
                self.chm_module_, 
                {torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d},
                dtype=torch.qint8)
            self.chm_module_.load_state_dict(ckpt, strict=False)
        else:
            ckpt = torch.load(ssl_path)
            state_dict = ckpt['state_dict']
            self.chm_module_.load_state_dict(state_dict)
        
        # Scale output by 10 to get canopy height
        self.chm_module = lambda x: 10 * self.chm_module_(x)

    def forward(self, x):
        x = self.chm_module(x)
        return x

class AerialImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None, size=256, overlap=64):
        self.image_dir = Path(image_dir)
        self.image_paths = (
            list(self.image_dir.glob('*.jpg')) +
            list(self.image_dir.glob('*.png')) +
            list(self.image_dir.glob('*.tif')) +
            list(self.image_dir.glob('*.jpeg'))
        )
        self.transform = transform
        self.size = size
        self.overlap = overlap
        self.step = self.size - self.overlap
        
        self.patches = []
        self.image_sizes = {}
        for img_path in self.image_paths:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            self.image_sizes[str(img_path)] = (img_width, img_height)
            
            x_positions = list(range(0, img_width - self.size + 1, self.step))
            y_positions = list(range(0, img_height - self.size + 1, self.step))
            
            for y in y_positions:
                for x in x_positions:
                    self.patches.append((img_path, x, y))

    def __len__(self):
        return len(self.patches)
        
    def __getitem__(self, idx):
        img_path, x, y = self.patches[idx]
        img = Image.open(img_path).convert('RGB')
        img_patch = img.crop((x, y, x+self.size, y+self.size))
        if self.transform:
            img_patch = self.transform(img_patch)
        else:
            img_patch = T.ToTensor()(img_patch)
        return {
            'img': img_patch, 
            'img_path': str(img_path), 
            'x_coord': x, 
            'y_coord': y
        }

def evaluate(model, norm, model_norm, preprocessed_dir, bs=32, trained_rgb=False, normtype=2, device='cuda:0', display=False, image_dir='./data/images/', overlap=64):
    print("normtype", normtype)    
    ds = AerialImageDataset(image_dir=image_dir, size=256, overlap=overlap)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
        
    preprocessed_path = Path(preprocessed_dir)
    output_images_dir = preprocessed_path / 'predicted_canopy_heights'
    output_images_dir.mkdir(parents=True, exist_ok=True)

    predictions_per_image = {}
    for ipath in ds.image_paths:
        p_str = str(ipath)
        W, H = ds.image_sizes[p_str]
        predictions_per_image[p_str] = {
            'width': W,
            'height': H,
            'sum_array': np.zeros((H, W), dtype=np.float32),
            'weight_sum_array': np.zeros((H, W), dtype=np.float32)
        }

    model.eval()

    # Create a spatial weight mask
    tile_size = 256
    # Example: Gaussian weighting mask that peaks at center and tapers at edges.
    center = tile_size // 2
    sigma = overlap / 2.0  # Adjust sigma as needed for smoother transitions
    y_coords, x_coords = np.ogrid[:tile_size, :tile_size]
    dist_sq = (x_coords - center)**2 + (y_coords - center)**2
    weight_mask = np.exp(-dist_sq / (2.0 * sigma * sigma))
    weight_mask /= weight_mask.max()  # normalize to 1 at center

    # Convert weight_mask to torch if you want to do on GPU or stay on CPU if blending on CPU
    # Here we can keep it on CPU for simplicity.
    weight_mask = weight_mask.astype(np.float32)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch['img'].to(device)
            img_paths = batch['img_path']
            xs = batch['x_coord'].cpu().numpy()
            ys = batch['y_coord'].cpu().numpy()
            
            img_norm = norm(img)
            
            pred = model(img_norm)
            pred = pred.cpu().detach().relu().numpy()  # shape: (N,1,256,256)
            
            for i in range(pred.shape[0]):
                p_str = img_paths[i]
                x = xs[i]
                y = ys[i]
                pred_img = pred[i,0,:,:]  # a 256x256 patch
                
                # Apply weighted blending
                # Multiply patch by weight_mask before adding
                predictions_per_image[p_str]['sum_array'][y:y+256, x:x+256] += pred_img * weight_mask
                predictions_per_image[p_str]['weight_sum_array'][y:y+256, x:x+256] += weight_mask

    # After processing all patches, compute weighted average
    for p_str, data in predictions_per_image.items():
        sum_array = data['sum_array']
        weight_sum_array = data['weight_sum_array']
        # Avoid division by zero just in case
        weight_sum_array[weight_sum_array == 0] = 1
        out_array = sum_array / weight_sum_array
        out_image = Image.fromarray(out_array, mode='F')
        base_name = Path(p_str).stem
        tif_output_path = output_images_dir / f"{base_name}_pred.tif"
        out_image.save(tif_output_path)
        
    print(f"Prediction images saved under {output_images_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run canopy height inference on aerial images.')
    parser.add_argument('--checkpoint', type=str, help='CHM pred checkpoint file', default='saved_checkpoints/compressed_SSLlarge.pth')
    parser.add_argument('--preprocessed_dir', type=str, required=True, help='Directory created during preprocessing for a specific NAIP image. Predictions will be placed here.')
    parser.add_argument('--trained_rgb', action='store_true', help='Set if model was finetuned on aerial data')
    parser.add_argument('--normnet', type=str, help='Path to a normalization network', default='saved_checkpoints/aerial_normalization_quantiles_predictor.ckpt')
    parser.add_argument('--normtype', type=int, help='0: no norm; 1: old norm; 2: new norm', default=2) 
    parser.add_argument('--display', action='store_true', help='If set, save additional PNG visualizations (not required).')
    parser.add_argument('--image_dir', type=str, help='Directory containing input aerial tiles', default='./data/images/')
    parser.add_argument('--overlap', type=int, help='Overlap in pixels for sliding window patches', default=64)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if 'compressed' in args.checkpoint:
        device = 'cpu'
    else:
        device = 'cuda:0'
    
    # Load normalization network
    norm_path = args.normnet 
    ckpt = torch.load(norm_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    for k in list(state_dict.keys()):
        if 'backbone.' in k:
            new_k = k.replace('backbone.', '')
            state_dict[new_k] = state_dict.pop(k)
    
    model_norm = RNet(n_classes=6)
    model_norm = model_norm.eval()
    model_norm.load_state_dict(state_dict)
        
    # Load SSL model
    model = SSLModule(ssl_path=args.checkpoint)
    model.to(device)
    model = model.eval()
    
    # Image normalization
    norm = T.Normalize((0.420, 0.411, 0.296), (0.213, 0.156, 0.143))
    norm = norm.to(device)
    
    evaluate(
        model,
        norm,
        model_norm,
        preprocessed_dir=args.preprocessed_dir,
        bs=16,
        trained_rgb=args.trained_rgb,
        normtype=args.normtype,
        device=device,
        display=args.display,
        image_dir=args.image_dir,
        overlap=args.overlap
    )

if __name__ == '__main__':
    main()
