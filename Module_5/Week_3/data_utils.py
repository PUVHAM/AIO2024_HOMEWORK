import os
import gdown
import zipfile
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize 
from torchvision.io import read_image
from Module_5.Week_3.config import DatasetConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
img_height, img_width = (128, 128)
    
class ImageDataset(Dataset):
    def __init__(self, img_dir, norm, label2idx, 
                 split='train', train_ratio=0.8):
        self.resize = Resize((img_height, img_width))
        self.norm = norm
        self.split = split
        self.train_ratio = train_ratio
        self.img_dir = img_dir
        self.label2idx = label2idx

        self.img_paths, self.img_labels = self.read_img_files()

        if split in ['train', 'val'] and 'train' in img_dir.lower():
            train_data, val_data = train_test_split(
                list(zip(self.img_paths, self.img_labels)),
                train_size=self.train_ratio,
                random_state=DatasetConfig.RANDOM_SEED,
                stratify=self.img_labels
            )

            if split == 'train':
                self.img_paths, self.img_labels = zip(*train_data)
            elif split == 'val':
                self.img_paths, self.img_labels = zip(*val_data)

    def read_img_files(self):
        img_paths = []
        img_labels = []
        for cls in self.label2idx.keys():
            for img in os.listdir(os.path.join(self.img_dir, cls)):
                img_paths.append(os.path.join(self.img_dir, cls, img))
                img_labels.append(cls)

        return img_paths, img_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        cls = self.img_labels[idx]
        img = self.resize(read_image(img_path))
        img = img.type(torch.float32)
        label = self.label2idx[cls]

        if self.norm:
            img = (img / 127.5) - 1

        return img, label

def download_dataset(csv_name):
    dataset_ids = {
        "Regression": "1qiUDDoYyRLBiKOoYWdFl_5WByHE8Cugu",  
        "Classification": "1SqSn_8rxkk-Qvu4JLMcN_3ZFGDNa6P_V",
        "Image_Classification": "1GaTMURqIQTjtalbNVAyVgPIEis21A0r8"
    }
    
    file_id = dataset_ids[csv_name]
    url = f'https://drive.google.com/uc?id={file_id}'
    
    os.makedirs(DatasetConfig.DATASET_DIR, exist_ok=True)
    
    output_path = DatasetConfig.DATASET_PATH[csv_name]
    
    gdown.download(url, output=output_path, quiet=True, fuzzy=True)
    print(f"Downloaded {csv_name} dataset to {output_path}")
    
    if csv_name == "Image_Classification": 
        extract_to = os.path.splitext(output_path)[0] 
        _unzip_file(output_path, extract_to) 

def _unzip_file(zip_path, extract_to):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The zip file '{zip_path}' does not exist.")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted '{zip_path}' to '{extract_to}'")

def load_dataset(file_name):
    file_path = DatasetConfig.DATASET_PATH[file_name]
    
    if not os.path.exists(file_path):
        try:
            download_dataset(file_name)
        except Exception as e:
            ERROR_MSG = f"Failed to download the {file_name} dataset. Please check the download process."
            raise e(ERROR_MSG) 

    if file_name == "Regression":
        data = pd.read_csv(file_path)
        X = data.drop(columns=['MPG']).values
        y = data['MPG'].values
        return _split_and_build_data_loader(X, y, file_name)
    elif file_name == "Classification":
        data = np.load(file_path, allow_pickle=True).item()
        X, y = data["X"], data["labels"]
        return _split_and_build_data_loader(X, y, file_name)
    elif file_name == "Image_Classification":
        return _build_image_data_loader()
    else:
        raise ValueError(f"Unsupported {file_name}. Only 'Regression', 'Classification', and 'Image_Classification' are allowed.")

def _split_and_build_data_loader(x, y, file_name):

    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      test_size=DatasetConfig.VAL_SIZE,
                                                      random_state=DatasetConfig.RANDOM_SEED,
                                                      shuffle=DatasetConfig.IS_SHUFFLE)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                        test_size=DatasetConfig.TEST_SIZE,
                                                        random_state=DatasetConfig.RANDOM_SEED,
                                                        shuffle=DatasetConfig.IS_SHUFFLE)

    scaler = StandardScaler()
    x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
    x_val = torch.tensor(scaler.transform(x_val), dtype=torch.float32)
    x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
    
    dtype = torch.float32 if file_name == "Regression" else torch.long
    y_train = torch.tensor(y_train, dtype=dtype)
    y_val = torch.tensor(y_val, dtype=dtype)
    y_test = torch.tensor(y_test, dtype=dtype)

    train_loader = DataLoader(CustomDataset(x_train, y_train), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(CustomDataset(x_val, y_val), batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(CustomDataset(x_test, y_test), batch_size=32, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def _build_image_data_loader():
    train_dir = DatasetConfig.IMAGE_CLASSIFICATION_DIR["Train"]
    test_dir = DatasetConfig.IMAGE_CLASSIFICATION_DIR["Test"]
    
    classes = os.listdir(train_dir)
    label2idx = {cls: idx for idx, cls in enumerate(classes)}

    train_loader = DataLoader(
        ImageDataset(train_dir, norm=True, label2idx=label2idx, split='train'),
        batch_size=256, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        ImageDataset(train_dir, norm=True, label2idx=label2idx, split='val'),
        batch_size=256, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        ImageDataset(test_dir, norm=True, label2idx=label2idx, split='test'),
        batch_size=256, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_dataset("Regression")
    print(next(iter(train_loader))[0].shape)
    train_loader, val_loader, test_loader = load_dataset("Classification")
    print(len(set(d[1].item() for d in train_loader.dataset)))
    download_dataset("Image_Classification")