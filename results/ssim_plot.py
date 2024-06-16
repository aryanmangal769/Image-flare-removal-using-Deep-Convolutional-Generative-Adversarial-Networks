# Usage:
#
# python3 script.py --input original.png --output modified.png
# Based on: https://github.com/mostafaGwely/Structural-Similarity-Index-SSIM-

# 1. Import the necessary packages
from skimage.metrics import structural_similarity
import argparse
import imutils
import cv2
import os 
import matplotlib.pyplot as plt

root_path = "evaluation_images"
orig_image = ["input_0.png", "input_50.png", "input_99.png", "input_150.png", "input_195.png"]
tran_image = ["y_gen_0.png", "y_gen_50.png", "y_gen_99.png", "y_gen_150.png", "y_gen_195.png"]

ssim_lis = []

for oi, ti in zip(orig_image[::-1], tran_image[::-1]):
    oi_path = os.path.join(root_path, oi)
    ti_path = os.path.join(root_path, ti)

    imageA = cv2.imread(oi_path)
    imageB = cv2.imread(ti_path)

    # 4. Convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 5. Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    # 6. You can print only the score if you want
    ssim_lis.append(score)

ssim_lis.sort()
plt.plot([1, 50, 100, 150, 200], ssim_lis, marker = "x", color = "black", mfc = "red", mec="red")
plt.xlabel("Number of Epochs")
plt.ylabel("SSIM Score")
plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
plt.show()

