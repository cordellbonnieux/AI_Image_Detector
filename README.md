# Group 14 - AI Image Detector

## use virtual environment
source venv/bin/activate

## install dependencies
python -m pip install -r requirements.txt

## train model
python -m src.training.train

## predict single image
python -m src.app.predict path/to/image.jpg

## predict many images
python -m src.app.predict image1.jpg image2.jpg image3.jpg

## start REST API server
python -m src.app.app

## evaluate model accuracy & generate confusion matrix
python -m src.utils.evaluate