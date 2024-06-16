# Image-flare-removal-using-Deep-Convolutional-Generative-Adversarial-Networks
This repository contains the code for the paper: [Image Flare Removal Using Deep Convolutional Generative Adversarial Networks](https://ieeexplore.ieee.org/abstract/document/10116874).

# Introduction
In this paper, a novel approach is proposed for removing flare artefacts from images using a two-stage method. Flare is a phenomenon that occurs when a strong source of light hits the lens of the camera. The converging light rays are scattered due to imperfections in the lens, creating a glare or halo effect in the image. This degrades the image quality and can make it challenging to extract useful information from the image, making it problematic for industries such as autonomous driving and medical robotics that rely heavily on image processing algorithms. To solve this problem, we utilize a flare removal model based on a conditional generative network. This model is trained on a dataset created using computer vision techniques. The dataset creation process involves scraping internet images of different scenes under various lighting conditions without the presence of flare. These images are then pre-processed to extract the flare-corrupted images, which serve as input for the flare removal model, which produces flare-free output. However, in order to make the model more generalizable and able to handle real-world data, we incorporate the flare removal module in a self-supervised setting along with a flare addition module. This allows us to fine-tune the flare removal model using real images and improve its performance. The flare removal module achieves an SSIM score of 0.916, which was previously 0.887. Our approach can be used in a variety of applications, such as autonomous driving and camera photo optimization, and has the potential to be further improved by incorporating a flare mask detector module.

# Setup repo

### Create python environment
```
python venv env
```
### Activate environment
```
./env/bin/activate
```
### Clone repo and install requirements
```
git clone 
pip install -r requirements
```


## Code to run file

```
!python train.py --data_path /content/gdrive/MyDrive/flare_removal --save_model --load_model --gen_path path_to_saved_gen_model --disc_path path_to_saved_disc_model --n_epochs 1
```
