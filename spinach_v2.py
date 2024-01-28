import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import PIL

from transformers import pipeline

classifier = pipeline("image-classification", model="mhgun/leafer")

QUARTER_CM2 = 4.62244

# Find the stupid round boi
def find_quarter(masks):
    for i, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask['segmentation'].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        perimeter = cv2.arcLength(contours[0], True)
        area = cv2.contourArea(contours[0])
        if perimeter == 0:
            continue

        circularity = 4*np.pi*(area/(perimeter**2))

        if circularity > 0.85 and mask['area'] > 1000:
            return i, mask

def img_to_area(img_path):
    # Load image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Scale image
    max_dim = max(image.shape)
    max_dim_idx = image.shape.index(max_dim)
    target = 800
    if max_dim_idx == 0:
        ry = int((target/image.shape[0]) * image.shape[1])
        smol_img = cv2.resize(image, (ry, target))
    elif max_dim_idx == 1:
        rx = int((target/image.shape[1]) * image.shape[0])
        smol_img = cv2.resize(image, (target, rx))

    # Generate masks
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # mask_generator = SamAutomaticMaskGenerator(sam)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.95,
        stability_score_thresh=0.97,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        # min_mask_region_area=640000,  # Requires open-cv to run post-processing
    )
    masks = mask_generator.generate(smol_img)
    # Find quarter
    idx, quarter_mask = find_quarter(masks)
    del masks[idx]

    areas = []
    for i, mask in enumerate(masks):
        # Skip masks that are obviously too big or too small
        if mask['area'] > 1000 and mask['area'] < 100000:
            m = mask['segmentation']
            masked_img = smol_img.copy()
            masked_img[~m] = (0,0,0)
            pil_img = PIL.Image.fromarray(masked_img)
            # Run classifier to check if it looks leafy
            if classifier(pil_img)[0]['label'] == 'leaf':
                areas.append(mask['area'] * (QUARTER_CM2/quarter_mask['area']))
                # You can save the masked images to files with this line
                # pil_img.save(f"leaf{i}.png")
            
    return areas