import os

from PIL import Image

import torch.utils.data as data

class VGGFace2Dataset(data.Dataset):
    def __init__(self, name, data_dir, transform=None):
        self.name = name
        self.transform = transform
        
        id_label_map = {}
        for index, id in enumerate(os.listdir(data_dir)):
            id_label_map[id] = index

        self.image_paths = []
        self.labels = []
        for index, id in enumerate(os.listdir(data_dir)):
            root_id_dir = os.path.join(data_dir + id)
            self.image_paths += [os.path.join(root_id_dir, img) for img in os.listdir(root_id_dir)]
            self.labels += [id_label_map[id] for img in os.listdir(root_id_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Train labels and val labels DO NOT reference the same class. Be cautious.
        label = self.labels[idx]

        return image, label