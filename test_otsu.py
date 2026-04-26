import cv2
import glob
import os
import numpy as np

file_paths = glob.glob("dataset/5/*.jpg")
if not file_paths:
    print("No images found in dataset/5/")
else:
    for path in file_paths[:1]:
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Test both
        _, binary1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((7, 7), np.uint8)
        cleaned1 = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel)
        cleaned1 = cv2.morphologyEx(cleaned1, cv2.MORPH_OPEN, kernel)
        
        cleaned2 = cv2.morphologyEx(binary2, cv2.MORPH_CLOSE, kernel)
        cleaned2 = cv2.morphologyEx(cleaned2, cv2.MORPH_OPEN, kernel)
        
        print("Binary 1 (Normal): ", "White pixels:", np.sum(cleaned1 == 255), "Black pixels:", np.sum(cleaned1 == 0))
        print("Binary 2 (Inv)   : ", "White pixels:", np.sum(cleaned2 == 255), "Black pixels:", np.sum(cleaned2 == 0))
