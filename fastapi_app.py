from fastapi import FastAPI, UploadFile, File
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import io
from network_model import transform, Net # network_model.py から前処理とネットワークの定義を読み込み

app = FastAPI()

# ネットワークの準備
net = Net().cpu().eval()
# 学習済みモデルの重み（blood.pt）を読み込み
net.load_state_dict(torch.load('blood.pt', map_location=torch.device('cpu')))

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # 画像の読み込み
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')

    # 画像の前処理
    image = transform(image)
    image = image.unsqueeze(0) # 1次元増やす

    # 推論
    prediction = net(image)[0]

    # 結果の抽出
    boxes = prediction['boxes'].tolist()
    labels = prediction['labels'].tolist()
    scores = prediction['scores'].tolist()

    return {
        "boxes": boxes,
        "labels": labels,
        "scores": scores
    }
