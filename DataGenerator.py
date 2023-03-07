from torch.utils.data import Dataset
import pandas as pd
import os
import torch

class DataGeneration(Dataset):
    def __init__(self, DIR_name, image_size, train=True):
        self.DIR_name = DIR_name
        self.train = train
        self.image_size = image_size
        try:
            self.data = pd.read_csv(os.path.join(
                DIR_name, f"{'train' if train else 'test'}.csv"))
        except:
            raise FileNotFoundError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = (self.data.iloc[idx, 1:].values).reshape(self.image_size,
                                                        self.image_size).astype('float32')
        class_id = self.data.iloc[idx, 0]
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        class_id = torch.tensor([class_id]).squeeze()
        return img_tensor, class_id