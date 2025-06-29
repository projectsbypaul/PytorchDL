import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import gc
from typing import List, Tuple, Dict, Optional

class InteractiveDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None, class_dict: Optional[Dict] = None, transform=None, set_name: str = "default"):
        if data.dim() == 4:
            data = data.unsqueeze(1)  # Add channel dim if missing [B, D, H, W] -> [B, 1, D, H, W]
        self.data = data.clone().detach()
        self.labels = labels.clone().detach() if labels is not None else None
        if self.labels is not None and self.labels.dim() == 3:
            self.labels = self.labels.squeeze_(1)

        self.transform = transform
        self.__class_dict = class_dict
        self.__split_ratio = 0.8
        self.batch_size = 32
        self.__name = set_name

        self.__train_dataset = None
        self.__test_dataset = None
        self.__train_loader = None
        self.__test_loader = None

        self.split_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        label = self.labels[idx] if self.labels is not None else None
        return sample, label

    def split_dataset(self):
        train_size = int(self.__split_ratio * len(self.data))
        test_size = len(self.data) - train_size
        self.__train_dataset, self.__test_dataset = random_split(self, [train_size, test_size])
        self.__train_loader = None
        self.__test_loader = None

    def get_train_loader(self, batch_size: int) -> DataLoader:
        if self.__train_loader is None or self.__train_loader.batch_size != batch_size:
            self.__train_loader = DataLoader(self.__train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        return self.__train_loader

    def get_test_loader(self, batch_size: int) -> DataLoader:
        if self.__test_loader is None or self.__test_loader.batch_size != batch_size:
            self.__test_loader = DataLoader(self.__test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        return self.__test_loader

    def get_train_dataset(self):
        return self.__train_dataset

    def get_test_dataset(self):
        return self.__test_dataset

    def get_class_dictionary(self):
        return self.__class_dict

    def get_name(self):
        return self.__name

    def set_split(self, split: float):
        self.__split_ratio = split

    def set_name(self, name: str):
        self.__name = name

    def save_dataset(self, file_path):
        save_dict = {"data": self.data, "labels": self.labels, "class_dict": self.__class_dict}
        torch.save(save_dict, file_path)
        print(f"Dataset saved to {file_path}")

    def get_info(self) -> str:
        info = (f"Dataset: {self.__name}\n"
                f"Samples: {len(self.data)}\n"
                f"Data shape: {self.data.shape} - dtype: {self.data[0].dtype}\n"
                f"Labels shape: {self.labels.shape if self.labels is not None else 'None'}\n"
                f"Class dictionary: {self.__class_dict}")
        return info

    @staticmethod
    def load_dataset(file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No dataset found at {file_path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loaded_data = torch.load(file_path, map_location="cpu", weights_only=False)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        data = loaded_data["data"]
        labels = loaded_data.get("labels", None)
        class_dict = loaded_data.get("class_dict", None)

        return InteractiveDataset(data, labels, class_dict)