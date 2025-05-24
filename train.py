import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import sys

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# Data paths - adjust these for your system
BASE_DIR = r"C:\Users\felme\Downloads\ml_exercise_therapanacea\data"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_img")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val_img")
TRAIN_LABEL_PATH = os.path.join(BASE_DIR, "label_train.txt")
OUTPUT_LABEL_PATH = os.path.join(BASE_DIR, "label_val.txt")

# Custom Dataset class with proper image-label mapping
class ImageDataset(Dataset):
    def __init__(self, img_dir, label_file=None, transform=None, minority_transform=None, is_test=False):
        self.img_dir = img_dir
        self.transform = transform
        self.minority_transform = minority_transform
        self.is_test = is_test
        
        # Sort images numerically by filename
        self.img_paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )
        
        # Load labels for training data
        self.all_labels = None
        if label_file and not is_test:
            try:
                # Try pandas first
                labels_df = pd.read_csv(label_file, sep="\t", header=None)
                self.all_labels = labels_df[0].values
            except:
                try:
                    labels_df = pd.read_csv(label_file, header=None)
                    self.all_labels = labels_df[0].values
                except:
                    # Fallback to basic file reading
                    with open(label_file, 'r') as f:
                        self.all_labels = [int(line.strip()) for line in f.readlines()]
        
        print(f"Loaded {len(self.img_paths)} images from {os.path.basename(img_dir)}")
        if self.all_labels is not None:
            print(f"Loaded {len(self.all_labels)} labels")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.is_test or self.all_labels is None:
            # Test data - return image and filename
            if self.transform:
                image = self.transform(image)
            img_name = os.path.basename(img_path)
            return image, img_name
        else:
            # Training data - map image number to label
            img_name = os.path.basename(img_path)
            img_number = int(img_name.split('.')[0])
            label_idx = img_number - 1  # Convert to 0-based index
            
            if label_idx >= len(self.all_labels) or label_idx < 0:
                raise IndexError(f"Image {img_name} -> label index {label_idx} out of bounds")
            
            label = self.all_labels[label_idx]
            
            # Apply transforms - extra augmentation for minority class
            if label == 0 and self.minority_transform:
                image = self.minority_transform(image)
            elif self.transform:
                image = self.transform(image)
            
            return image, label

# Test the dataset mapping
def test_dataset_mapping():
    print("Testing dataset mapping...")
    
    test_dataset = ImageDataset(TRAIN_IMG_DIR, TRAIN_LABEL_PATH, is_test=False)
    
    # Check first few mappings
    print("First 10 mappings:")
    for i in range(min(10, len(test_dataset))):
        try:
            image, label = test_dataset[i]
            img_path = test_dataset.img_paths[i]
            img_name = os.path.basename(img_path)
            img_number = int(img_name.split('.')[0])
            print(f"  {img_name} (#{img_number}) -> Label {label}")
        except Exception as e:
            print(f"  Error at index {i}: {e}")
    
    # Test some random samples
    print("\nRandom samples:")
    random_indices = random.sample(range(len(test_dataset)), min(5, len(test_dataset)))
    for i in random_indices:
        try:
            image, label = test_dataset[i]
            img_path = test_dataset.img_paths[i]
            img_name = os.path.basename(img_path)
            print(f"  {img_name} -> Label {label}")
        except Exception as e:
            print(f"  Error at index {i}: {e}")
    
    print("Dataset mapping test completed.\n")

def check_setup():
    """Verify data setup"""
    print("Checking data setup...")
    
    train_images = glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg"))
    test_images = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg"))
    
    print(f"Training images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")
    
    if os.path.exists(TRAIN_LABEL_PATH):
        with open(TRAIN_LABEL_PATH, 'r') as f:
            labels = f.readlines()
        print(f"Training labels: {len(labels)}")
    else:
        print("ERROR: No training labels found!")
        return False
    
    print("Setup verified - ready to proceed.\n")
    return True

# Data transforms
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_minority_transforms():
    """More aggressive augmentation for minority class"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# CNN Model
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

# Create weighted sampler for imbalanced data
def create_weighted_sampler(dataset, indices=None):
    print("Creating weighted sampler for class imbalance...")
    
    if indices is not None:
        # For subset (train/val split)
        labels = []
        for idx in indices:
            img_path = dataset.img_paths[idx]
            img_name = os.path.basename(img_path)
            img_number = int(img_name.split('.')[0])
            label_idx = img_number - 1
            labels.append(dataset.all_labels[label_idx])
    else:
        # For full dataset
        labels = []
        for idx in range(len(dataset)):
            img_path = dataset.img_paths[idx]
            img_name = os.path.basename(img_path)
            img_number = int(img_name.split('.')[0])
            label_idx = img_number - 1
            labels.append(dataset.all_labels[label_idx])
    
    class_counts = Counter(labels)
    print(f"Class distribution: {class_counts}")
    
    # Calculate sample weights
    weights = [1.0 / class_counts[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True
    )
    
    return sampler

# Training function - using fixed 0.5 threshold to prevent data leakage
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10):
    """Train model with validation"""
    best_val_loss = float('inf')
    best_model_weights = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []
        
        if val_loader is not None:
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # Calculate metrics using fixed 0.5 threshold
            val_pred = (np.array(all_probs) >= 0.5).astype(int)
            hter, far, frr = calculate_hter(all_labels, val_pred)
            
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = val_correct / val_total
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            
            if scheduler is not None:
                scheduler.step(epoch_val_loss)
            
            # Save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_weights = model.state_dict().copy()
                print(f'New best model saved! Val Loss: {epoch_val_loss:.4f}')
        else:
            epoch_train_loss = train_loss / len(train_loader.dataset)
            if epoch_train_loss < best_val_loss:
                best_val_loss = epoch_train_loss
                best_model_weights = model.state_dict().copy()
                print('New best model saved!')
        
        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        if val_loader is not None:
            print(f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
            print(f'  HTER: {hter:.4f}, FAR: {far:.4f}, FRR: {frr:.4f}')
        print()
    
    # Load best weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print("Best model weights loaded.")
    
    return model, history

# Prediction functions
def predict_test_set(model, test_loader, device, threshold=0.5):
    """Generate predictions for test set"""
    print(f"Generating predictions with threshold = {threshold}...")
    
    model.eval()
    predictions = []
    img_names = []
    probabilities = []
    
    with torch.no_grad():
        for inputs, names in tqdm(test_loader, desc='Predicting'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= threshold).long()
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            img_names.extend(names)
    
    # Sort by image name
    name_to_pred = dict(zip(img_names, predictions))
    name_to_prob = dict(zip(img_names, probabilities))
    
    sorted_names = sorted(name_to_pred.keys(), key=lambda x: int(x.split('.')[0]))
    sorted_predictions = [name_to_pred[name] for name in sorted_names]
    sorted_probabilities = [name_to_prob[name] for name in sorted_names]
    
    return sorted_predictions, sorted_names, sorted_probabilities

def get_prediction_probs(model, data_loader, device):
    """Get prediction probabilities for analysis"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Getting probabilities'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_probs), np.array(all_labels)

def save_predictions(predictions, output_path):
    """Save predictions to file"""
    print(f"Saving predictions to {output_path}...")
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print("Predictions saved.")

def calculate_hter(y_true, y_pred):
    """Calculate Half-Total Error Rate"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn)  # False Acceptance Rate
    frr = fn / (fn + tp)  # False Rejection Rate
    hter = (far + frr) / 2  # Half Total Error Rate
    return hter, far, frr

# Visualization functions
def plot_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='train', marker='o')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='validation', marker='s')
    axes[0].set_title('Training History - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='train', marker='o')
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(history['val_acc'], label='validation', marker='s')
    axes[1].set_title('Training History - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=['Class 0', 'Class 1']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_scores):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curve(y_true, y_scores):
    """Plot precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve (AP = {average_precision:.2f})')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()

def analyze_dataset():
    """Analyze dataset for class distribution"""
    print("Analyzing dataset...")
    
    # Read labels
    try:
        labels_df = pd.read_csv(TRAIN_LABEL_PATH, sep="\t", header=None)
        labels = labels_df[0].values
    except:
        try:
            labels_df = pd.read_csv(TRAIN_LABEL_PATH, header=None)
            labels = labels_df[0].values
        except:
            with open(TRAIN_LABEL_PATH, 'r') as f:
                labels = [int(line.strip()) for line in f.readlines()]
    
    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"Class distribution: {distribution}")
    
    # Calculate class weights
    total = len(labels)
    class_weights = {cls: total / (len(distribution) * count) for cls, count in distribution.items()}
    print(f"Class weights: {class_weights}")
    
    return class_weights, labels

def main():
    print("CNN Binary Classifier")
    print("=" * 50)
    print("Note: Using fixed 0.5 threshold to prevent data leakage")
    print("=" * 50)
    
    start_time = time.time()
    
    # Setup verification
    if not check_setup():
        return
    
    # Test dataset mapping
    test_dataset_mapping()
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Analyze dataset
    class_weights, all_labels = analyze_dataset()
    
    # Setup transforms
    train_transform = get_transforms(is_train=True)
    minority_transform = get_minority_transforms()
    test_transform = get_transforms(is_train=False)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ImageDataset(
        TRAIN_IMG_DIR, 
        TRAIN_LABEL_PATH, 
        transform=train_transform,
        minority_transform=minority_transform
    )
    
    # Train/validation split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Training: {train_size} images")
    print(f"Validation: {val_size} images")
    
    # Create data loaders
    train_sampler = create_weighted_sampler(train_dataset, train_subset.indices)
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=64, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Test dataset
    test_dataset = ImageDataset(VAL_IMG_DIR, transform=test_transform, is_test=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Test: {len(test_dataset)} images")
    
    # Initialize model
    print("\nInitializing model...")
    model = CustomCNN(num_classes=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup training
    # Use class weights if imbalanced
    if max(class_weights.values()) / min(class_weights.values()) > 1.2:
        print("Using weighted loss for class imbalance")
        weights = torch.FloatTensor([class_weights[0], class_weights[1]]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Train with validation
    print("\nTraining model...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20
    )
    
    # Plot results
    print("\nGenerating plots...")
    plot_history(history)
    
    # Validation analysis
    print("Analyzing validation performance...")
    val_probs, val_true = get_prediction_probs(model, val_loader, device)
    
    plot_roc_curve(val_true, val_probs)
    
    # Using fixed 0.5 threshold
    val_preds = (val_probs >= 0.5).astype(int)
    
    hter, far, frr = calculate_hter(val_true, val_preds)
    print(f"\nValidation Metrics (threshold = 0.5):")
    print(f"HTER: {hter:.4f}")
    print(f"FAR: {far:.4f}")
    print(f"FRR: {frr:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(val_true, val_preds))
    
    plot_confusion_matrix(val_true, val_preds)
    plot_precision_recall_curve(val_true, val_probs)
    
    # Retrain on full dataset
    print("\nRetraining on full training set...")
    
    full_train_dataset = ImageDataset(
        TRAIN_IMG_DIR, 
        TRAIN_LABEL_PATH, 
        transform=train_transform,
        minority_transform=minority_transform
    )
    
    full_train_sampler = create_weighted_sampler(full_train_dataset)
    full_train_loader = DataLoader(
        full_train_dataset, 
        batch_size=64, 
        sampler=full_train_sampler,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Reset model
    model = CustomCNN(num_classes=2).to(device)
    
    if max(class_weights.values()) / min(class_weights.values()) > 1.2:
        weights = torch.FloatTensor([class_weights[0], class_weights[1]]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    # Final training
    model, _ = train_model(
        model, full_train_loader, None, criterion, optimizer, None, device, epochs=20
    )
    
    # Save model
    print("\nSaving model...")
    model_path = os.path.join(BASE_DIR, 'cnn_classifier.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Generate final predictions
    print("\nGenerating final predictions...")
    
    # Use fixed 0.5 threshold
    test_predictions, test_names, test_probs = predict_test_set(
        model, test_loader, device, threshold=0.5
    )
    
    print(f"Test images: {len(test_names)}")
    print(f"Predictions: {len(test_predictions)}")
    
    pred_counts = dict(zip(*np.unique(test_predictions, return_counts=True)))
    print(f"Prediction distribution: {pred_counts}")
    
    # Save predictions
    save_predictions(test_predictions, OUTPUT_LABEL_PATH)
    
    # Save probabilities
    prob_path = os.path.join(BASE_DIR, 'val_probabilities.txt')
    with open(prob_path, 'w') as f:
        for prob in test_probs:
            f.write(f"{prob:.6f}\n")
    print(f"Probabilities saved to {prob_path}")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Final validation HTER: {hter:.3f}")
    print(f"Predictions saved to: {OUTPUT_LABEL_PATH}")
    print("\nReady for submission!")

if __name__ == "__main__":
    main()