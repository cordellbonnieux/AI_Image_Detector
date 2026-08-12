# Group 14 - AI Image Detector

## using virtual environment: 
source venv/bin/activate

## train:
python -m src.training.train

## Using the predictor:

### Single image
python -m src.app.predict path/to/image.jpg

### With verbose output
python -m src.app.predict path/to/image.jpg -v

### Multiple images
python -m src.app.predict image1.jpg image2.jpg image3.jpg

### Flask API
python -m src.app.app

## Evaluate model
python -m src.utils.evaluate