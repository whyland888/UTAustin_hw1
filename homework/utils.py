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

        # Images
        image_files = pd.read_csv(os.path.join(dataset_path, 'labels.csv'))['file'].tolist()
        image_paths = [os.path.join(dataset_path, x) for x in image_files]
        transform = transforms.Compose([transforms.ToTensor()])
        self.images = [Image.open(x).convert("RGB") for x in image_paths]
        self.image_tensors = [transform(x) for x in self.images]

        # Labels
        self.str_labels = pd.read_csv(os.path.join(dataset_path, 'labels.csv'))['label'].tolist()
        # print(self.str_labels[0:10])
        label_dict = {"background": 0, "kart": 1, "pickup": 2, "nitro": 3, "bomb": 4, "projectile": 5}
        self.int_labels = [label_dict[x] for x in self.str_labels]

    def __len__(self):
        return len(self.int_labels)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        image = self.image_tensors[idx]
        label = self.int_labels[idx]

        return image, label


def load_data(dataset_path, num_workers=24, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


# PATH = r"/home/bojangles/Desktop/UT_Austin_NLP/UTAustin_hw1/data/train"
#
# ds = SuperTuxDataset(dataset_path=PATH)
# print(ds[0][0].shape)


