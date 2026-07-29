# src/app/predict.py

import torch
import os
from PIL import Image
from src.models.ai_detector import AIDetectorCNN
from src.utils.preprocess import test_transform
from src.config import CHECKPOINT_DIR, DEVICE, NUM_CLASSES, NUM_CHANNELS

class AIDetectorPredictor:
    def __init__(self, model_path=None):
        self.device = DEVICE
        
        # Initialize model
        self.model = AIDetectorCNN(in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
        self.model.to(self.device)
        
        # Load trained weights
        if model_path is None:
            model_path = os.path.join(CHECKPOINT_DIR, "ai_detector.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set to evaluation mode
        
        # Class labels
        self.classes = ["AI Generated", "Not AI Generated"]
    
    def predict_image(self, image_path):
        # Load and preprocess the image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Open image and apply transforms
        image = Image.open(image_path).convert('RGB')
        image_tensor = test_transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.classes[predicted.item()]
        confidence_score = confidence.item() * 100
        
        return predicted_class, confidence_score, probabilities.cpu().numpy()
    
    def predict_batch(self, image_paths):
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                results.append((image_path, *result))
            except Exception as e:
                results.append((image_path, f"Error: {str(e)}", 0, None))
        return results


def main():
    #Command-line interface for prediction.
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict if an image is AI generated")
    parser.add_argument("image_path", nargs="+", help="Path(s) to image file(s)")
    parser.add_argument("--model_path", help="Path to model checkpoint", default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = AIDetectorPredictor(model_path=args.model_path)
    
    # Predict
    if len(args.image_path) == 1:
        # Single image
        try:
            class_name, confidence, probs = predictor.predict_image(args.image_path[0])
            print(f"\nImage: {args.image_path[0]}")
            print(f"Prediction: {class_name}")
            print(f"Confidence: {confidence:.2f}%")
            if args.verbose:
                print(f"Probabilities: Not AI: {probs[0][0]*100:.2f}%, AI: {probs[0][1]*100:.2f}%")
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Multiple images
        results = predictor.predict_batch(args.image_path)
        for result in results:
            if len(result) == 4:
                path, class_name, confidence, probs = result
                print(f"\n{path}:")
                print(f"  Prediction: {class_name}")
                print(f"  Confidence: {confidence:.2f}%")
                if args.verbose:
                    print(f"  Probabilities: Not AI: {probs[0][0]*100:.2f}%, AI: {probs[0][1]*100:.2f}%")
            else:
                print(f"\n{result[0]}: {result[1]}")

if __name__ == "__main__":
    main()