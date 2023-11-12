# 必要なモジュールのインポート
from torchvision import transforms
# import pytorch_lightning as pl
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 前処理
transform = transforms.Compose([
    transforms.ToTensor()
])

class Net(nn.Module):

    def __init__(self, n_feature=1024, n_class=4):
        super().__init__()

        self.faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(n_feature, n_class)


    def forward(self, x, t=None):
        if self.training:
            return self.faster_rcnn(x, t)
        else:
            return self.faster_rcnn(x)