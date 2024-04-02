from torch.utils.data import Dataset
import os
from PIL import Image

class PlantTraitsDataset(Dataset):
    def __init__(self, img_dir, X_img, X_data, y, transform=None):
        self.img_dir = img_dir
        self.img_df = X_img
        self.data_df = X_data
        self.labels_df = y
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_df.iloc[idx, 0].split('/')[1])
        image = Image.open(img_name).convert('RGB')
        data = self.data_df
        labels = y

        if self.transform:
            image = self.transform(image)

        return image, data, labels