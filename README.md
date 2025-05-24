# CNN Binary Classifier

A PyTorch implementation for binary image classification.

## What it does

Trains a CNN to classify images into two categories. The main focus was on:
- Handling class imbalance properly
- Preventing data leakage during validation
- Getting the image-label mapping right (surprisingly tricky!)

## Dataset Exploration

Before jumping into the model, I did some exploration to understand what we're working with. Check out `exploratory_analysis.ipynb` - it covers:

- Dataset overview (100k training + 20k test images)
- Class distribution analysis (found some imbalance issues)
- Image properties and size analysis
- Image-label mapping verification (this was crucial!)
- Visual samples from each class
- Key findings that guided the model design

The notebook shows the whole analytical process from "what do we have?" to "what should we do about it?"

## Requirements

```
pip install torch torchvision scikit-learn matplotlib seaborn pandas numpy pillow tqdm jupyter
```

Or use the requirements file:
```
pip install -r requirements.txt
```

## Quick setup

1. Put your data in this structure:
```
data/
├── train_img/          # training images
├── val_img/            # test images  
└── label_train.txt     # labels for training
```

2. Run training:
```
python train.py
```

The script will output predictions to `data/label_val.txt`.

## Model details

Pretty standard CNN with 4 conv blocks:
- 32 → 64 → 128 → 256 filters
- BatchNorm + ReLU + MaxPool
- Dropout in classifier (0.5)
- About 2M parameters

Nothing fancy, just solid fundamentals.

## Key technical decisions

**Fixed 0.5 threshold**: I use 0.5 throughout validation instead of optimizing the threshold. This prevents data leakage but might hurt performance slightly. Better to be conservative.

**Class imbalance handling**: 
- Weighted sampling during training
- Class-weighted loss function
- Extra augmentation for minority class
- (The EDA revealed the extent of the imbalance problem)

**Data integrity**: Added lots of checks for the image-label mapping. Had some bugs early on where images weren't matching their labels correctly. The exploratory analysis helped catch this early.

## Results

Gets decent performance on the validation set. Check the generated plots in the results folder for detailed metrics.

The main outputs:
- `label_val.txt` - final predictions
- Training plots and confusion matrix
- Model checkpoint

## Code structure

```
├── exploratory_analysis.ipynb  # data exploration and analysis
├── train.py                   # main training script
├── requirements.txt           # dependencies
├── data/                     # put your dataset here
└── results/                  # outputs go here
```

Kept it simple - everything you need is in train.py. Could split it into modules but honestly this works fine for the project size. The notebook shows my analytical process before diving into the model.

## Notes

- Uses CUDA if available
- Seeds are fixed for reproducibility  
- Saves best model based on validation loss
- All the standard metrics: accuracy, HTER, ROC curves etc.

Built this over a weekend, so it's not over-engineered but gets the job done reliably.

## License

MIT - do whatever you want with it.