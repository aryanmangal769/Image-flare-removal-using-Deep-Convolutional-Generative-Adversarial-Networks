import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from options import get_args
import os 
opt = get_args()

data_source = opt.data_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = os.path.join(data_source, "train") 
VAL_DIR = os.path.join(data_source, "val")
LEARNING_RATE = opt.learning_rate
BATCH_SIZE = opt.batch_size
NUM_WORKERS = 2
IMAGE_SIZE = opt.load_size
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = opt.n_epochs
LOAD_MODEL = opt.load_model
SAVE_MODEL = opt.save_model
CHECKPOINT_DISC = opt.disc_path
CHECKPOINT_GEN = opt.gen_path
LOSS_FILE = "results.txt"
GEN_SAVE_NAME = "gen.pth.tar"
DISC_SAVE_NAME = "disc.pth.tar"


info = CHECKPOINT_DISC.split("_")
if len(info) > 1:
    START_EPOCH = int(info[0])
else:
    START_EPOCH = 0


both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
