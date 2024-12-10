def evaluate(model, norm, model_norm, preprocessed_dir, bs=32, trained_rgb=False, normtype=2, device='cuda:0', display=False, image_dir='./data/images/'):
    print("normtype", normtype)    
    ds = AerialImageDataset(image_dir=image_dir, patch_size=256, stride=256)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
        
    preprocessed_path = Path(preprocessed_dir)
    output_images_dir = preprocessed_path / 'reassembled_canopy_heights'
    output_images_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        print(f"Number of batches in dataloader: {len(dataloader)}")
        for batch in tqdm(dataloader):
            patches = batch[0]
            if isinstance(patches, list) and len(patches) > 0:
                print(f"Number of patches: {len(patches)}")
                img_path = patches[0]['img_path']
                img_width = patches[0]['img_width']
                img_height = patches[0]['img_height']
                
                reassembled_image = np.zeros((img_height, img_width), dtype=np.float32)
                
                for patch in patches:
                    img = patch['img'].unsqueeze(0).to(device)
                    x_idx, y_idx = patch['x_idx'], patch['y_idx']
                    
                    img_norm = norm(img)
                    pred = model(img_norm)
                    pred = pred.cpu().detach().relu()
                    
                    if pred.numel() == 0:
                        print(f"Empty prediction for patch {x_idx}, {y_idx}")
                        continue
                    
                    pred_img = pred[0][0].numpy()
                    h, w = pred_img.shape
                    print(f"pred_img shape: {pred_img.shape}, x_idx: {x_idx}, y_idx: {y_idx}, h: {h}, w: {w}")
                    
                    if h == 0 or w == 0:
                        print(f"Invalid dimensions for pred_img: h={h}, w={w}")
                        continue
                    
                    print(f"Prediction min: {pred_img.min()}, max: {pred_img.max()}, mean: {pred_img.mean()}")
                    reassembled_image[y_idx:y_idx+h, x_idx:x_idx+w] = pred_img

                reassembled_image_pil = Image.fromarray(reassembled_image.astype('float32'), mode='F')
                output_path = output_images_dir / (Path(img_path).stem + '_reassembled.tif')
                print(f"Saving reassembled image to {output_path}")
                reassembled_image_pil.save(output_path)
