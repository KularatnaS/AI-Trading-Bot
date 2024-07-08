# LLM-from-scratch
## How to set up
Create a python 3.8 virtual environment and install the project:
```
python3.8 -m venv venv
source venv/bin/activate
pip install -e .
```
## How to train
Set training parameters at `config.config.py`, place training and validation data in .txt format at 'data/train' 
and 'data/val' respectively, and run:
```
python train.py
```

## How to perform inference
Provide a prompt and a path to a trained model as shown in `inference.py` and run:
```
python inference.py
```