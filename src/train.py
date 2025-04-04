import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

from data.dataset import get_data_loaders
from models.model import MyModel

#compute the class weights for the training set
#the class weights are computed based on the inverse frequency of the classes in the training set
def compute_class_weights(train_loader, device):
    class_counts = torch.zeros(101, device=device)  # 0-99 for numbers, 100 for no-number
    total_samples = 0
    
    # Count samples for each class
    for _, labels in train_loader:
        labels = labels.to(device)
        for label in labels:
            class_counts[label] += 1
            total_samples += 1
    
    # Calculate base weights (inverse frequency)
    class_weights = total_samples / (class_counts * len(class_counts))
    class_weights[class_counts == 0] = 0  # Handle classes with zero samples
    
    # Apply moderate boost to no-number class (100)
    no_number_weight = 1.5  # Moderate boost factor
    class_weights[100] *= no_number_weight
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum()
    
    return class_weights

#train the model

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    no_number_correct = 0
    no_number_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        # Calculate regular accuracy
        mask = labels != 100  # Regular numbers
        if mask.sum() > 0:
            total += mask.sum().item()
            correct += predicted[mask].eq(labels[mask]).sum().item()
        
        # Calculate no-number accuracy
        no_number_mask = labels == 100  # No-number cases
        if no_number_mask.sum() > 0:
            no_number_total += no_number_mask.sum().item()
            no_number_correct += predicted[no_number_mask].eq(labels[no_number_mask]).sum().item()
    
    acc = 100. * correct / total if total > 0 else 0
    no_number_acc = 100. * no_number_correct / no_number_total if no_number_total > 0 else 0
    return running_loss / len(train_loader), acc, no_number_acc

#validate the model

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    no_number_correct = 0
    no_number_total = 0
    
    #disable gradient calculation for validation
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # Calculate regular accuracy
            mask = labels != 100  # Regular numbers
            if mask.sum() > 0:
                total += mask.sum().item()
                correct += predicted[mask].eq(labels[mask]).sum().item()
            
            # Calculate no-number accuracy
            no_number_mask = labels == 100  # No-number cases
            if no_number_mask.sum() > 0:
                no_number_total += no_number_mask.sum().item()
                no_number_correct += predicted[no_number_mask].eq(labels[no_number_mask]).sum().item()
    
    acc = 100. * correct / total if total > 0 else 0
    no_number_acc = 100. * no_number_correct / no_number_total if no_number_total > 0 else 0
    return running_loss / len(val_loader), acc, no_number_acc

def main():
    # Parameters
    num_epochs = 35 
    batch_size = 64  
    initial_lr = 0.001
    warmup_epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data paths
    csv_path = "src/data_label.csv"
    image_folder = "src/train&valdata"
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        csv_path, 
        image_folder,
        batch_size=batch_size
    )
    
    # Initialize model
    model = MyModel(num_classes=100).to(device)  # 100 classes for numbers 00-99
    
    # Compute class weights for balanced loss
    class_weights = compute_class_weights(train_loader, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Initialize optimizer with warmup
    optimizer = optim.Adam(
        model.parameters(),
        lr=initial_lr/10,  # Start with lower learning rate
        weight_decay=1e-5,  # Lighter regularization
        betas=(0.9, 0.999)  # Default Adam betas
    )
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs)
    
    # Training history
    history = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Warmup period
        if epoch < warmup_epochs:
            lr = initial_lr/10 + (initial_lr - initial_lr/10) * epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
        
        train_loss, train_acc, train_no_number_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_no_number_acc = validate(model, val_loader, criterion, device)
        
        # Save metrics
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_no_number_accuracy': train_no_number_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_no_number_accuracy': val_no_number_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, No-Number: {train_no_number_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, No-Number: {val_no_number_acc:.2f}%')
        print('-' * 50)
        
        # Save history to CSV
        pd.DataFrame(history).to_csv('training_history3.csv', index=False)
        
        # Save model if it's the best validation accuracy so far
        if epoch == 0 or val_acc > max(h['val_accuracy'] for h in history[:-1]):
            torch.save(model.state_dict(), 'best_model3.pth')

if __name__ == '__main__':
    main()


