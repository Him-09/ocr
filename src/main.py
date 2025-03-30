import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from data.dataset import get_data_loaders
from models.model import SimpleModel
from training.trainer import train, evaluate, save_predictions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)


def validate_paths(csv_path, image_folder, csv_path1, image_folder1):
    """Validate all required paths exist and are accessible."""
    paths = {
        'Train CSV': csv_path,
        'Train Images': image_folder,
        'Test CSV': csv_path1,
        'Test Images': image_folder1
    }

    for name, path in paths.items():
        if not path:
            raise ValueError(f"{name} path is empty. Please set the path.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} path does not exist: {path}")
        if name.endswith('CSV') and not path.endswith('.csv'):
            logging.warning(f"{name} path should end with .csv: {path}")


def check_dataset_consistency(csv_path, image_folder):
    """Check if all images referenced in CSV exist in the image folder."""
    import pandas as pd
    missing_images = []

    df = pd.read_csv(csv_path)
    for img_name in df.iloc[:, 0]:
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            missing_images.append(img_name)

    if missing_images:
        logging.warning(f"Found {len(missing_images)} missing images in {image_folder}")
        logging.warning("First 5 missing images: " + ", ".join(missing_images[:5]))
        return False
    return True


def main():
    # Set your dataset paths here
    csv_path = "data_label.csv"
    image_folder = "train&valdata"
    csv_path1 = "test1_data_label.csv"
    image_folder1 = "testdata"

    try:
        # Validate paths
        validate_paths(csv_path, image_folder, csv_path1, image_folder1)

        # Check dataset consistency
        if not check_dataset_consistency(csv_path, image_folder):
            logging.warning("Dataset consistency check failed. Some images are missing.")
        if not check_dataset_consistency(csv_path1, image_folder1):
            logging.warning("Test dataset consistency check failed. Some images are missing.")

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {device}')

        # Get data loaders
        train_dataloader, val_dataloader, test_dataloader = get_data_loaders(
            csv_path, image_folder
        )
        test_dataloader = get_test_data_loaders(csv_path1, image_folder1)

        # Initialize model and move to device
        model = SimpleModel(100)  # 100 classes for numbers 00-99
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
        if os.path.exists('best_model1.pth'):
            checkpoint = torch.load('best_model1.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"\nBest model was saved at epoch {checkpoint['epoch'] + 1}")
            logging.info(f"Best validation loss: {checkpoint['val_loss']:.4f}")
        else:
            logging.warning("No saved model found. Using last trained model for evaluation.")

        # Final evaluation on test set
        evaluate(model, test_dataloader, 100, device)

        # Save predictions
        save_predictions(model, test_dataloader, device)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()