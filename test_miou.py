import os
import torch
import numpy as np
from PIL import Image

def calculate_iou(pred_mask, gt_mask):

    intersection = torch.logical_and(pred_mask, gt_mask).sum().float()
    union = torch.logical_or(pred_mask, gt_mask).sum().float()
    iou = intersection / union
    return iou.item()

def load_mask(file_path):
    mask = Image.open(file_path).convert("L") 
    mask = np.array(mask) 
    mask = torch.from_numpy(mask).float() 
    mask = mask > 0 
    return mask

def calculate_iou_for_directory(render_dir, gt_dir):
    iou_list = []

    for filename in os.listdir(render_dir):
        if filename.endswith(".png"): 
            render_path = os.path.join(render_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            pred_mask = load_mask(render_path)
            gt_mask = load_mask(gt_path)
            if gt_mask.sum() == 0:
                continue
            iou = calculate_iou(pred_mask, gt_mask)
            iou_list.append(iou)
    
    mean_iou = np.mean(iou_list) if iou_list else 0.0
    return mean_iou

render_dir=''
gt_dir=''
average_iou = calculate_iou_for_directory(render_dir, gt_dir)
print(f'iteration {i} Average IoU: {average_iou}')

