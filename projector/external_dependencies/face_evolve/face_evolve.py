import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from backbone.model_irse import IR_50

PWD = os.path.dirname(__file__)

INPUT_SIZE = [112, 112]
BACKBONE = IR_50(INPUT_SIZE)
BACKBONE.load_state_dict(torch.load(os.path.join(PWD, "backbone_ir50_ms1m_epoch120.pth")))
