from PIL import Image

import torch.utils.data as data

class VGGFace2ViSEDataset(data.Dataset):
    def __init__(self, dataframe, path_to_datset, transform=None):
        self.data = dataframe
        # self.data = self.data[:100_000]
        print(self.data.shape)
        
        self.data["image_1"] = path_to_datset + self.data["id_1"] + "/" + self.data["image_1"]
        self.data["image_2"] = path_to_datset + self.data["id_2"] + "/" + self.data["image_2"]
        self.data.drop(["id_1", "id_2"], axis=1, inplace=True)

        self.transform = transform

    def __len__(self):
        return len(self.data)//2
    
    def get_image_pair(self, idx):
        # Load images using torchvision.io.read_image with read_mode='RGB'
        img1_path = self.data.iloc[idx]['image_1']
        img2_path = self.data.iloc[idx]['image_2']

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # # Get labels
        # label = self.data.iloc[idx]['label']
        # label = -1 if label == 0 else label
        # label = torch.tensor(label, dtype=torch.float32)

        # Apply transformations if any
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            return img1, img2
        else:
            return img1, img2

    def __getitem__(self, idx):
        negative_pair_idx = idx*2

        neg_img1, neg_img2 = self.get_image_pair(negative_pair_idx)
        pos_img1, pos_img2 = self.get_image_pair(negative_pair_idx+1)

        return neg_img1, neg_img2, pos_img1, pos_img2