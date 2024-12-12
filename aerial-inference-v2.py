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
    def __init__(self, image_dir, transform=None, size=256):
        self.image_dir = Path(image_dir)
        self.image_paths = (
            list(self.image_dir.glob('*.jpg')) +
            list(self.image_dir.glob('*.png')) +
            list(self.image_dir.glob('*.tif')) +
            list(self.image_dir.glob('*.jpeg'))
        )
        self.transform = transform
        self.size = size
        
        # Precompute patches for all images and store image dimensions
        self.patches = []
        self.image_sizes = {}  # {str(path): (width, height)}
        for img_path in self.image_paths:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            self.image_sizes[str(img_path)] = (img_width, img_height)
            # Compute how many patches fit horizontally and vertically
            x_count = img_width // self.size
            y_count = img_height // self.size
            for yi in range(y_count):
                for xi in range(x_count):
                    x = xi * self.size
                    y = yi * self.size
                    self.patches.append((img_path, x, y))

    def __len__(self):
        return len(self.patches)
        
    def __getitem__(self, idx):
        img_path, x, y = self.patches[idx]
        img = Image.open(img_path).convert('RGB')
        # Crop the 256x256 patch at original resolution
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

def evaluate(model, norm, model_norm, preprocessed_dir, bs=32, trained_rgb=False, normtype=2, device='cuda:0', display=False, image_dir='./data/images/'):
    print("normtype", normtype)    
    ds = AerialImageDataset(image_dir=image_dir, size=256)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
        
    # Save predictions under preprocessed_dir/predicted_canopy_heights
    preprocessed_path = Path(preprocessed_dir)
    output_images_dir = preprocessed_path / 'predicted_canopy_heights'
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store assembled predictions for each image
    # predictions_per_image = {
    #   img_path_str: {
    #       'width': W, 'height': H,
    #       'array': np.zeros((H, W), dtype=float),
    #       'count_map': np.zeros((H, W), dtype=int) # if needed to handle overlaps
    #   }
    # }
    predictions_per_image = {}
    for ipath in ds.image_paths:
        p_str = str(ipath)
        W, H = ds.image_sizes[p_str]
        predictions_per_image[p_str] = {
            'width': W,
            'height': H,
            'array': np.zeros((H, W), dtype=np.float32)
        }
        
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch['img'].to(device)
            img_paths = batch['img_path']
            xs = batch['x_coord'].cpu().numpy()
            ys = batch['y_coord'].cpu().numpy()
            
            # Apply normalization
            img_norm = norm(img)
            
            # Model prediction
            pred = model(img_norm)
            pred = pred.cpu().detach().relu().numpy()  # shape: (N,1,256,256)
            
            for i in range(pred.shape[0]):
                p_str = img_paths[i]
                x = xs[i]
                y = ys[i]
                pred_img = pred[i,0,:,:]  # a 256x256 patch
                
                # Insert the patch predictions back into the large array
                predictions_per_image[p_str]['array'][y:y+256, x:x+256] = pred_img

    # After processing all patches for all images, save the assembled images
    for p_str, data in predictions_per_image.items():
        out_array = data['array']
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
    
    # Evaluate using patch-based inference and then assemble
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
        image_dir=args.image_dir
    )

if __name__ == '__main__':
    main()
