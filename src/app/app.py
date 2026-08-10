from flask import Flask, request, jsonify
import os
import io
import torch
from PIL import Image
from src.models.ai_detector import AIDetectorCNN
from src.utils.preprocess import test_transform
from src.config import CHECKPOINT_DIR, DEVICE

app = Flask(__name__)

device = DEVICE
model_path = os.path.join(CHECKPOINT_DIR, "ai_detector.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

model = AIDetectorCNN(use_pretrained=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

class_names = ["AI Generated", "Not AI Generated"]


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    return test_transform(image).unsqueeze(0).to(device)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model_path": model_path})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use the 'image' field."}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Unable to read image: {exc}"}), 400

    with torch.no_grad():
        tensor = preprocess_image(image)
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_index = int(torch.argmax(outputs, dim=1).item())

    return jsonify({
        "prediction": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index] * 100),
        "probabilities": {
            "ai_generated": float(probabilities[0] * 100),
            "not_ai_generated": float(probabilities[1] * 100),
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
