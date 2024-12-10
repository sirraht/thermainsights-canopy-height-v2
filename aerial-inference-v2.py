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
        
        self.chm_module = lambda x: 10 * self.chm_module_(x)

    def forward(self, x):
        x = self.chm_module(x)
        return x

class AerialImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None, patch_size=256, stride=256):
        self.image_dir = Path(image_dir)
        self.image_paths = (
            list(self.image_dir.glob('*.jpg')) +
            list(self.image_dir.glob('*.png')) +
            list(self.image_dir.glob('*.tif')) +
            list(self.image_dir.glob('*.jpeg'))
        )
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size

        patches = []
        for y in range(0, img_height, self.stride):
            for x in range(0, img_width, self.stride):
                crop_box = (x, y, x + self.patch_size, y + self.patch_size)
                patch = img.crop(crop_box)
                
                if self.transform:
                    patch = self.transform(patch)
                else:
                    patch = T.ToTensor()(patch)
                
                patches.append({'img': patch, 'img_path': str(img_path), 'x_idx': x, 'y_idx': y, 'img_width': img_width, 'img_height': img_height})
                
        return patches

def evaluate(model, norm, model_norm, preprocessed_dir, bs=32, trained_rgb=False, normtype=2, device='cuda:0', display=False, image_dir='./data/images/'):
    print("normtype", normtype)    
    ds = AerialImageDataset(image_dir=image_dir, patch_size=256, stride=256)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
        
    preprocessed_path = Path(preprocessed_dir)
    output_images_dir = preprocessed_path / 'reassembled_canopy_heights'
    output_images_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            patches = batch[0]
            img_path = patches[0]['img_path']
            img_width = patches[0]['img_width']
            img_height = patches[0]['img_height']
            
            reassembled_image = np.zeros((img_height, img_width), dtype=np.float32)
            
            for patch in patches:
                img = patch['img'].unsqueeze(0).to(device)  # Add batch dimension
                x_idx, y_idx = patch['x_idx'], patch['y_idx']
                
                img_norm = norm(img)
                pred = model(img_norm)
                pred = pred.cpu().detach().relu()
                
                pred_img = pred[0][0].numpy()
                h, w = pred_img.shape
                reassembled_image[y_idx:y_idx+h, x_idx:x_idx+w] = pred_img

            # Save the reassembled image as a single output file
            reassembled_image_pil = Image.fromarray(reassembled_image.astype('float32'), mode='F')
            output_path = output_images_dir / (Path(img_path).stem + '_reassembled.tif')
            reassembled_image_pil.save(output_path)

    print(f"Reassembled images saved under {output_images_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run canopy height inference on aerial images.')
    parser.add_argument('--checkpoint', type=str, help='CHM pred checkpoint file', default='saved_checkpoints/compressed_SSLlarge.pth')
    parser.add_argument('--preprocessed_dir', type=str, required=True, help='Directory for preprocessed predictions.')
    parser.add_argument('--trained_rgb', action='store_true', help='Set if model was finetuned on aerial data')
    parser.add_argument('--normnet', type=str, help='Path to normalization network', default='saved_checkpoints/aerial_normalization_quantiles_predictor.ckpt')
    parser.add_argument('--normtype', type=int, help='0: no norm; 1: old norm; 2: new norm', default=2) 
    parser.add_argument('--display', action='store_true', help='If set, save additional PNG visualizations.')
    parser.add_argument('--image_dir', type=str, help='Directory containing input aerial tiles', default='./data/images/')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = 'cuda:0' if 'compressed' not in args.checkpoint else 'cpu'
    
    ckpt = torch.load(args.normnet, map_location='cpu')
    state_dict = ckpt['state_dict']
    model_norm = RNet(n_classes=6)
    model_norm.load_state_dict(state_dict)
        
    model = SSLModule(ssl_path=args.checkpoint)
    model.to(device)
    model = model.eval()
    
    norm = T.Normalize((0.420, 0.411, 0.296), (0.213, 0.156, 0.143))
    norm = norm.to(device)
    
    evaluate(
        model, norm, model_norm,
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
