import os
import torch
from data.dataset import get_test_data_loaders
from models.model import MyModel

def evaluate(model, test_loader, device):
    """Evaluate the model on the test set."""
    model.eval()
    correct = 0
    total = 0
    no_number_correct = 0
    no_number_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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
    return acc, no_number_acc

def main():
    # Set paths
    csv_path = "src/test1_data_label.csv"
    image_folder = "src/testdata"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get test data loader
    test_loader = get_test_data_loaders(csv_path, image_folder)
    
    # Initialize model
    model = MyModel(100).to(device)
    
    # Load best model
    if os.path.exists('best_model.pth'):
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint)
        print("Loaded best model successfully")
    else:
        print("No saved model found. Please train the model first.")
        return
    
    # Evaluate on test set
    acc, no_number_acc = evaluate(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"Regular Accuracy: {acc:.2f}%")
    print(f"No-Number Accuracy: {no_number_acc:.2f}%")

if __name__ == '__main__':
    main()