import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, csv_path, image_folder, is_train=True):
        self.data = pd.read_csv(csv_path, delim_whitespace=True)
        self.data['label'] = self.data['label'].fillna(100).astype(int)
        self.image_folder = image_folder
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((64, 64)),
                transforms.RandomApply([
                    transforms.RandomRotation(3),
                    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03), shear=2)
                ], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self):
        return len(self.data)

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

def get_data_loaders(csv_path, image_folder, batch_size=64, train_split=0.8):
    train_dataset = MyDataset(csv_path, image_folder, is_train=True)
    val_dataset = MyDataset(csv_path, image_folder, is_train=False)
    
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def get_test_data_loaders(csv_path, image_folder, batch_size=64):
    test_dataset = MyDataset(csv_path, image_folder, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
