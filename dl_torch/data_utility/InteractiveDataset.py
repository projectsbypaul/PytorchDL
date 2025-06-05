import torch
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.utils.data import Dataset, DataLoader, random_split
import os
from typing import List, Tuple, Dict, Optional

class InteractiveDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None, class_dict: Optional[Dict] = None, transform=None, set_name:str = "default"):
        """
        Args:
            data (torch.Tensor): Tensor of 3D float arrays.
            labels (torch.Tensor, optional): Corresponding labels (if supervised).
            class_dict (dict, optional): Dictionary containing string conversion for labels.
            transform (callable, optional): Transformation function.
        """
        if data.dim() == 4:  # If missing channel dimension
            data = data.unsqueeze(1)  # Add a channel dimension
        self.data = data.clone().detach()  # Clone to avoid modifying original data
        self.__data_loader = None

        if labels.dim() == 3:
            labels = labels.squeeze_(1)
        self.labels = labels.clone().detach() if labels is not None else None

        self.__class_dict = class_dict if class_dict is not None else None
        self.transform = transform

        self.__split_ratio = 0.8  # Default train/test split ratio
        self.batch_size = 32  # Default batch size

        self.__name = set_name

        self.__train_loader = None
        self.__test_loader = None
        self.__train_dataset = None
        self.__test_dataset = None

        self.split_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        label = self.labels[idx] if self.labels is not None else torch.tensor([], dtype=torch.float32)
        return sample, label  # Always return a tuple

    def split_dataset(self):
        """Splits dataset into train and test sets."""
        train_size = int(self.__split_ratio * len(self.data))
        test_size = len(self.data) - train_size

        self.__train_dataset, self.__test_dataset = random_split(self, [train_size, test_size])

        self.__train_loader = DataLoader(self.__train_dataset, batch_size=self.batch_size, shuffle=True)
        self.__test_loader = DataLoader(self.__test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_train_loader(self, batch_size : int) -> DataLoader:

        self.__train_loader = DataLoader(self.__train_dataset, batch_size=batch_size, shuffle=True)

        return self.__train_loader

    def get_test_loader(self, batch_size : int) -> DataLoader:

        self.__test_loader = DataLoader(self.__test_dataset, batch_size=batch_size, shuffle=True)

        return self.__test_loader

    def get_data_loader(self, batch_size : int) -> DataLoader:

        self.__data_loader = DataLoader(self.data, batch_size=batch_size, shuffle=False)

        return self.__data_loader

    def get_class_dictionary(self):
        return self.__class_dict

    def get_name(self):
        return self.__name

    def set_split(self, split: float):
        self.__split_ratio = split

    def set_name(self, name: str):
        self.__name = name

    def save_dataset(self, file_path):
        """Save dataset to a file."""
        save_dict = {"data": self.data, "labels": self.labels, "class_dict": self.__class_dict}
        torch.save(save_dict, file_path)
        print(f"Dataset saved to {file_path}")

    def get_info(self) -> str:
        info = (f"Dataset: {self.__name}\n"
                f"Dataset with {len(self.data)} samples\n"
                f"data - {self.data.size()} of {self.data[0].dtype}\n"
                f"labels - {self.labels.size()} of {self.labels[0].dtype} \n"
                f"dictionary - {self.__class_dict}"
                )
        return info



    @staticmethod
    def load_dataset(file_path):
        """Load dataset from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No dataset found at {file_path}")

        loaded_data = torch.load(file_path, weights_only=False)
        data = loaded_data["data"]
        labels = loaded_data["labels"] if loaded_data["labels"] is not None else None
        class_dict : Dict = loaded_data["class_dict"] if loaded_data["class_dict"] is not None else None
        return InteractiveDataset(data, labels, class_dict)


def main() -> None:
    location = r"C:\Local_Data\ABC\ABC_torch\torch_data_ks_16_pad_4_bw_5_vs_adaptive_n2\00000002.torch"
    dataset = InteractiveDataset.load_dataset(location)
    print(dataset.get_info())


if __name__ == "__main__":
    main()


