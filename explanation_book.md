# License Plate Recognition Project - Detailed Explanation Book

## Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Detailed File Explanations](#detailed-file-explanations)
   - [main.py](#mainpy)
   - [dataset.py](#datasetpy)
   - [model.py](#modelpy)
   - [trainer.py](#trainerpy)
   - [train.py](#trainpy)
   - [evaluate.py](#evaluatepy)
4. [Training Process](#training-process)
5. [Evaluation Process](#evaluation-process)

## Project Overview
This project implements a deep learning model to recognize numbers in license plate images. The model can:
- Identify numbers from 0-99
- Detect when no number is present
- Handle various image conditions through data augmentation

## File Structure
```
src/
├── main.py           # Main entry point
├── data/
│   └── dataset.py    # Data loading and preprocessing
├── models/
│   └── model.py      # Neural network architecture
├── training/
│   └── trainer.py    # Training utilities
├── train.py          # Training script
└── evaluate.py       # Evaluation script
```

## Detailed File Explanations

### main.py
This is the main entry point of the application. Let's break it down:

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import get_data_loaders
from models.model import SimpleModel
from training.trainer import train, evaluate, save_predictions
```
- Imports necessary libraries:
  - `os`: For file operations
  - `torch`: PyTorch deep learning framework
  - `torch.nn`: Neural network modules
  - `torch.optim`: Optimization algorithms
  - Custom modules for data, model, and training

```python
def main():
    # Set your dataset paths here
    csv_path = "data_label.csv"
    image_folder = "train&valdata"
    csv_path1 = "test1_data_label.csv"
    image_folder1 = "testdata"
```
- Defines paths for training and test data
- `csv_path`: Contains labels for training data
- `image_folder`: Directory containing training images
- `csv_path1`: Contains labels for test data
- `image_folder1`: Directory containing test images

```python
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
```
- Checks if GPU is available, uses CPU if not
- Prints which device is being used

```python
        # Get data loaders
        train_dataloader, val_dataloader, test_dataloader = get_data_loaders(
            csv_path, image_folder
        )
        test_dataloader = get_test_data_loaders(csv_path1, image_folder1)
```
- Creates data loaders for training, validation, and test sets
- Data loaders handle batching and data loading efficiently

```python
        # Initialize model and move to device
        model = SimpleModel(100)  # 100 classes for numbers 00-99
        model = model.to(device)
```
- Creates the neural network model
- Moves model to appropriate device (GPU/CPU)

```python
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```
- Sets up loss function (CrossEntropyLoss for classification)
- Configures Adam optimizer with learning rate 0.001

```python
        # Define learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )
```
- Sets up learning rate scheduler
- Reduces learning rate when validation loss plateaus
- Helps in better convergence

```python
        # Training with validation
        train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, num_epochs=20,
                        device=device)
```
- Starts the training process
- Passes all necessary components to training function

```python
        # Load best model for evaluation
        if os.path.exists('best_model1.pth'):
            checkpoint = torch.load('best_model1.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nBest model was saved at epoch {checkpoint['epoch'] + 1}")
            print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
```
- Loads the best performing model
- Prints information about when it was saved

```python
        # Final evaluation on test set
        evaluate(model, test_dataloader, 100, device)
```
- Evaluates model performance on test set

```python
        # Save predictions
        save_predictions(model, test_dataloader, device)
```
- Saves model predictions for analysis

### dataset.py
This file handles all data loading and preprocessing. Let's break it down:

```python
class SimpleDataset(Dataset):
    def __init__(self, csv_path, image_folder, is_train=True):
        self.data = pd.read_csv(csv_path, delim_whitespace=True)
        self.data['label'] = self.data['label'].fillna(100).astype(int)
```
- Creates a custom dataset class
- Reads CSV file containing image names and labels
- Converts NaN values to 100 (no-number case)

```python
        self.image_folder = image_folder
        self.is_train = is_train
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((64, 64)),
                transforms.RandomApply([
                    transforms.RandomRotation(3),
                    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), 
                                          scale=(0.97, 1.03), shear=2)
                ], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
```
- Sets up data transformations
- For training:
  - Converts to grayscale
  - Resizes to 64x64
  - Applies random augmentations (rotation, translation, scaling)
  - Converts to tensor and normalizes

```python
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
```
- For validation/testing:
  - Same basic transformations but without augmentations

```python
    def __len__(self):
        return len(self.data)
```
- Returns total number of samples in dataset

```python
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_name']
        label = self.data.iloc[idx]['label']
        img_path = os.path.join(self.image_folder, img_name)
        
        try:
            img = Image.open(img_path)
            img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise
```
- Loads and transforms a single image
- Returns image and its label
- Includes error handling for missing images

```python
def get_data_loaders(csv_path, image_folder, batch_size=64, train_split=0.8):
    train_dataset = SimpleDataset(csv_path, image_folder, is_train=True)
    val_dataset = SimpleDataset(csv_path, image_folder, is_train=False)
    
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
```
- Creates data loaders for training and validation
- Splits data into train and validation sets
- Configures batch size and shuffling

### model.py
This file defines the neural network architecture:

```python
class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
```
- Defines convolutional layers:
  - First layer: 1 input channel (grayscale) → 64 output channels
  - Second layer: 64 → 128 channels
  - Third layer: 128 → 128 channels
- Each conv layer is followed by batch normalization

```python
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
```
- Max pooling for downsampling
- Dropout layers to prevent overfitting

```python
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes + 1)
```
- Fully connected layers:
  - First layer: 128*8*8 → 512 neurons
  - Output layer: 512 → num_classes+1 (for no-number case)

```python
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
```
- Forward pass implementation:
  1. Convolution + BatchNorm + ReLU
  2. Max pooling
  3. Repeat for all conv layers
  4. Dropout
  5. Flatten
  6. Fully connected layers with ReLU
  7. Final output layer

### trainer.py
This file contains training and evaluation utilities:

```python
def train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, num_epochs, device):
    best_val_loss = float('inf')
    history = []
```
- Initializes training variables
- Tracks best validation loss
- Stores training history

```python
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        no_number_correct = 0
        no_number_total = 0
```
- Training loop for each epoch
- Sets model to training mode
- Initializes metrics

```python
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels + 1)
            loss.backward()
            optimizer.step()
```
- Training step:
  1. Zero gradients
  2. Forward pass
  3. Calculate loss
  4. Backward pass
  5. Update weights

```python
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels + 1).sum().item()
```
- Calculates accuracy metrics
- Tracks correct predictions

```python
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
```
- Sets model to evaluation mode
- Initializes validation metrics

```python
        with torch.no_grad():
            for images, labels in val_dataloader:
                outputs = model(images)
                loss = criterion(outputs, labels + 1)
```
- Validation loop without gradient computation
- Calculates validation loss

```python
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': train_loss,
            }, 'best_model1.pth')
```
- Saves model when validation loss improves
- Stores training state

### train.py
This is the main training script:

```python
def compute_class_weights(train_loader, device):
    class_counts = torch.zeros(101, device=device)
    total_samples = 0
```
- Computes class weights for balanced loss
- Handles class imbalance

```python
    for _, labels in train_loader:
        labels = labels.to(device)
        for label in labels:
            class_counts[label] += 1
            total_samples += 1
```
- Counts samples for each class
- Calculates total samples

```python
    class_weights = total_samples / (class_counts * len(class_counts))
    class_weights[class_counts == 0] = 0
    no_number_weight = 1.5
    class_weights[100] *= no_number_weight
```
- Computes inverse frequency weights
- Applies special weight to no-number class

```python
def main():
    # Parameters
    num_epochs = 20
    batch_size = 128
    initial_lr = 0.001
    warmup_epochs = 3
```
- Sets training hyperparameters
- Configures batch size and learning rate

```python
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        csv_path, 
        image_folder,
        batch_size=batch_size
    )
```
- Creates data loaders for training

```python
    # Initialize model and optimizer
    model = SimpleModel(num_classes=100).to(device)
    class_weights = compute_class_weights(train_loader, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
```
- Sets up model and loss function
- Uses weighted loss for class balance

```python
    # Training loop
    for epoch in range(num_epochs):
        # Learning rate warmup
        if epoch < warmup_epochs:
            lr = initial_lr/5 + (initial_lr - initial_lr/5) * epoch / warmup_epochs
```
- Implements learning rate warmup
- Gradually increases learning rate

### evaluate.py
This file handles model evaluation:

```python
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    no_number_correct = 0
    no_number_total = 0
```
- Sets up evaluation metrics
- Tracks correct predictions

```python
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
```
- Evaluates model without gradient computation
- Gets predictions

```python
            mask = labels != 100  # Regular numbers
            if mask.sum() > 0:
                total += mask.sum().item()
                correct += predicted[mask].eq(labels[mask]).sum().item()
```
- Calculates accuracy for regular numbers
- Tracks correct predictions

```python
            no_number_mask = labels == 100  # No-number cases
            if no_number_mask.sum() > 0:
                no_number_total += no_number_mask.sum().item()
                no_number_correct += predicted[no_number_mask].eq(labels[no_number_mask]).sum().item()
```
- Calculates accuracy for no-number cases
- Tracks correct predictions

```python
def main():
    # Set paths and device
    csv_path = "src/test1_data_label.csv"
    image_folder = "src/testdata"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
- Sets up evaluation environment
- Configures paths and device

```python
    # Load model and evaluate
    model = SimpleModel(100).to(device)
    if os.path.exists('best_model.pth'):
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint)
```
- Loads trained model
- Prepares for evaluation

## Training Process
1. Data Preparation:
   - Load images and labels
   - Apply data augmentation
   - Create data loaders

2. Model Setup:
   - Initialize neural network
   - Set up loss function and optimizer
   - Configure learning rate scheduler

3. Training Loop:
   - Forward pass
   - Calculate loss
   - Backward pass
   - Update weights
   - Track metrics

4. Validation:
   - Evaluate on validation set
   - Save best model
   - Adjust learning rate

## Evaluation Process
1. Model Loading:
   - Load best trained model
   - Prepare for evaluation

2. Testing:
   - Run model on test set
   - Calculate accuracy metrics
   - Track no-number detection

3. Results:
   - Print evaluation metrics
   - Save predictions if needed 