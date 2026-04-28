import numpy as np
import cv2

def tps_to_heatmap(tps, crop, sigma=10):
    h, w = crop.shape[:2]
    heatmap = []
    yy, xx = np.mgrid[0:h, 0:w]
    for i, (x, y) in enumerate(tps):
        hm = np.zeros((h,w), dtype=np.float32)
        x = float(x)
        y = float(y)
        g = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        hm += g
        heatmap.append(hm)
    return np.array(heatmap).transpose(1, 2, 0)

def generate_overlay(crop, heatmaps):
    combined = np.max(heatmaps, axis=2)
    combined_uint8 = (np.clip(combined, 0, 1) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(combined_uint8, cv2.COLORMAP_JET)
    if heatmap_color.shape[:2] != crop.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(crop, 0.6, heatmap_color, 0.4, 0)
    return overlay
