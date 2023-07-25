from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        WARNING: Do not perform data normalization here. 
        """
        self.path = dataset_path
        self.image_files = [file for file in os.listdir(dataset_path) if file.endswith(".jpg")]
        self.labels = pd.read_csv(os.path.join(dataset_path, 'labels.csv'))['label'].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        image_path = os.path.join(self.path, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        return image, label


def load_data(dataset_path, num_workers=24, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


PATH = r"C:\Users\Will\OneDrive\Desktop\State Farm\UT Austin Deep Learning\homework1\data\train"
ds = SuperTuxDataset(PATH)

