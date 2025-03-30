import os
import torch
import logging
from src.data.dataset import get_test_data_loaders
from src.models.model import SimpleModel

def validate_paths(csv_path, image_folder):
    """Validate all required paths exist and are accessible."""
    paths = {
        'Test CSV': csv_path,
        'Test Images': image_folder
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

    df = pd.read_csv(csv_path, delim_whitespace=True)
    for img_name in df['image_name']:
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            missing_images.append(img_name)

    if missing_images:
        logging.warning(f"Found {len(missing_images)} missing images in {image_folder}")
        logging.warning("First 5 missing images: " + ", ".join(missing_images[:5]))
        return False
    return True

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
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set your dataset paths here
    csv_path = "src/test1_data_label.csv"
    image_folder = "src/testdata"

    try:
        # Validate paths
        validate_paths(csv_path, image_folder)

        # Check dataset consistency
        if not check_dataset_consistency(csv_path, image_folder):
            logging.warning("Dataset consistency check failed. Some images are missing.")

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {device}')

        # Get test data loader
        test_loader = get_test_data_loaders(csv_path, image_folder)

        # Initialize model and move to device
        model = SimpleModel(100)  # 100 classes for numbers 00-99
        model = model.to(device)

        # Load the best model
        if os.path.exists('best_model2.pth'):
            checkpoint = torch.load('best_model2.pth')
            model.load_state_dict(checkpoint)
            logging.info("Loaded best model successfully")
        else:
            logging.error("No saved model found. Please train the model first.")
            return

        # Evaluate on test set
        acc, no_number_acc = evaluate(model, test_loader, device)
        
        logging.info(f"\nTest Results:")
        logging.info(f"Regular Accuracy: {acc:.2f}%")
        logging.info(f"No-Number Accuracy: {no_number_acc:.2f}%")

        logging.info("Evaluation completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()