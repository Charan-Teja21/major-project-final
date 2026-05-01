import numpy as np
from PIL import Image
import cv2
import os

def check_modality(image_path, expected_type):
    try:
        image = Image.open(image_path).convert('L') # Convert to grayscale
        img_array = np.array(image)
        
        bright_pixel_ratio = np.sum(img_array > 200) / img_array.size
        is_ct_likely = bright_pixel_ratio > 0.005
        
        print(f"File: {os.path.basename(image_path)}")
        print(f"  Expected: {expected_type}")
        print(f"  Bright pixel ratio: {bright_pixel_ratio:.4%}")
        print(f"  Is CT likely: {is_ct_likely}")
        
        if expected_type == 'CT':
            if not is_ct_likely:
                return False, f"Uploaded image lacks typical CT high-intensity bone signal (Found {bright_pixel_ratio:.1%} bright pixels). It looks like an MRI."
        elif expected_type == 'MRI':
            if is_ct_likely:
                return False, f"Uploaded image has high-intensity regions typical of CT scans (Found {bright_pixel_ratio:.1%} bright pixels). It looks like a CT."
                
        return True, ""
    except Exception as e:
        return True, str(e)

base_path = r"c:\Users\Administrator\Downloads\C578-20260220T110908Z-3-001\C578\C578-20260216T110029Z-3-001\C578\MRI-CT Brain Tumor Detection\MRI-CT Brain Tumor Detection"

test_files = [
    ("MRI_no-tumor.png", "MRI"),
    ("MRI_no_tumor2.png", "MRI"),
    ("CT_no_tumor.png", "CT"),
    ("CT_no_tumor2.png", "CT"),
    ("MRI_tumor.png", "MRI"),
    ("CT_tumor.png", "CT")
]

for filename, expected in test_files:
    path = os.path.join(base_path, filename)
    if os.path.exists(path):
        res, msg = check_modality(path, expected)
        print(f"  Result: {'Valid' if res else 'Invalid'} - {msg}")
        print("-" * 40)
    else:
        print(f"File not found: {filename}")
