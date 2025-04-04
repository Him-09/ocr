import os
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import get_data_loaders
from models.model import MyModel

def main():
    # Set your dataset paths here
    csv_path = "data_label.csv"
    image_folder = "train&valdata"
    csv_path1 = "test1_data_label.csv"
    image_folder1 = "testdata"

    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        # Get data loaders
        train_dataloader, val_dataloader = get_data_loaders(
            csv_path, image_folder
        )
        test_dataloader = get_test_data_loaders(csv_path1, image_folder1)

        # Initialize model and move to device
        model = MyModel(100)  # 100 classes for numbers 00-99
        model = model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Define learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )

        # Training with validation
        train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, num_epochs=20,
                        device=device)

        # Load best model for evaluation
        if os.path.exists('best_model.pth'):
            checkpoint = torch.load('best_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
           # print(f"\nBest model was saved at epoch {checkpoint['epoch'] + 1}")
           # print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
        else:
            print("No saved model found. Using last trained model for evaluation.")

        # Final evaluation on test set
        evaluate(model, test_dataloader, 100, device)


    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()
