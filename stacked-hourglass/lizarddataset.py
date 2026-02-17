class LizardDataset(torch.utils.data.Dataset):
    def __init__(self, npz_paths, aug_factor=8):
        self.paths = npz_paths
        self.aug_factor = aug_factor
        self.heatmap_size = 128
    def __len__(self):
        return len(self.paths) * self.aug_factor

    def __getitem__(self, idx):
        true_idx = idx % len(self.paths)
        data = np.load(self.paths[true_idx])
        img = data['image']
        heatmaps = data['heatmap']
        img, heatmaps = apply_base_transform(img, heatmaps)
        img, heatmaps = apply_augmentation(img, heatmaps)
        
        img_tensor = torch.from_numpy(img).float() / 255.0
        heatmaps_tensor = torch.from_numpy(heatmaps).permute(2,0,1).float()  # C,H,W
        heatmaps_tensor = F.interpolate(
            heatmaps_tensor.unsqueeze(0),
            size=(128,128),
            mode='bilinear',
            align_corners=False
        )
        return img_tensor, heatmaps_tensor