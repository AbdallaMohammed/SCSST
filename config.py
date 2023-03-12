import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_DIR = './checkpoints'
U2NET_SALIENCY_MAP_CHECKPOINT_FILE = 'u2net.pth'
U2NET_CLOTHES_CHECKPOINT_FILE = 'cloth_segm_u2net_latest.pth'

ENABLE_STYLE_TRANSFER = False
IMG_SIZE = 356
EPOCHS = 6000
LEARNING_RATE = 0.001
ALPHA = 1
BETA = 0.01
EPOCH_CHECKPOINT = 200

basic_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(
        mean=(.5, .5, .5),
        std=(.5, .5, .5),
    ),
    ToTensorV2()
])
