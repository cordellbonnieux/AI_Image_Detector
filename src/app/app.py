from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
from src.utils.preprocess import preprocess_image

app = Flask(__name__)

model = torch.load("models/ai_detector.pth")
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    output = model(preprocess_image(img))
    return jsonify({"likelihood": float(output.item())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)