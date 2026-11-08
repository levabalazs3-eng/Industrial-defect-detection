import io
import torch
import uvicorn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from model import CustomDefectCNN

app = FastAPI(title = "Industrial Defect Detection API")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomDefectCNN().to(device)
model.load_state_dict(torch.load("defect_cnn_model.pth", map_location=device, weights_only=True))
model.eval()

def process_image(image_bytes: bytes) -> torch.Tensor:
    img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    return transform(img_pil).unsqueeze(0).to(device)

@app.post("/predict")
async def predict_defect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = process_image(image_bytes)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        class_names = ['defective', 'good']
        pred_label = class_names[prediction.item()]

        return {
            "filename": file.filename,
            "prediction": pred_label.upper(),
            "confidence": round(float(confidence.item()), 4),
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)