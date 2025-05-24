# Setup Guide

## Getting this running

1. Clone the repo:
```bash
git clone https://github.com/yourusername/cnn-classifier.git
cd cnn-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your data folder:
```
data/
├── train_img/     # put training images here
├── val_img/       # put test images here  
└── label_train.txt # training labels (one per line)
```

4. Update the paths in train.py if needed (line ~25):
```python
BASE_DIR = r"path/to/your/data"
```

5. Run training:
```bash
python train.py
```

## What happens during training

- Splits training data 90/10 for train/validation
- Handles class imbalance with weighted sampling
- Trains for 20 epochs with early stopping
- Saves plots and model checkpoint
- Retrains on full dataset for final model
- Generates predictions for test set

## Outputs

- `data/label_val.txt` - predictions for your test images
- `results/` folder with plots and model file
- Console output with training progress and metrics

## Troubleshooting

**CUDA out of memory**: Reduce batch size in the script (currently 64)

**Image-label mismatch errors**: Check that your image filenames are numbers (000001.jpg, 000002.jpg, etc.) and match the order in label_train.txt

**Poor performance**: The dataset might be very imbalanced. Check the class distribution output early in training.

That's pretty much it. The script is fairly robust and should handle most edge cases.