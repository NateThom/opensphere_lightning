import random
import os.path as osp

import numpy as np
import cv2

from skimage import transform

from torch.utils.data import Dataset

class ClassDataset(Dataset):
    def __init__(self, name, data_dir, ann_path,
            test_mode=False, noise_ratio=None, seed=None):
        super().__init__()

        self.name = name
        self.data_dir = data_dir
        self.ann_path = ann_path
        self.test_mode = test_mode
        self.noise_ratio = noise_ratio
        self.seed = seed

        self.get_data()
        self.get_label()

    def get_data(self):
        """Get data from a provided annotation file.
        """
        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

        self.data_items = []
        for line in lines:
            path, name= line.rstrip().split()
            item = {'path': path, 'name': name}
            self.data_items.append(item)

        if len(self.data_items) == 0:
            raise (RuntimeError('Found 0 files.'))

    def corrupt_label(self):
        random.seed(self.seed)
        labels = list({item['label'] for item in self.label_items})
        for item in self.label_items:
            if random.random() > self.noise_ratio:
                continue
            item['label'] = random.choice(labels)

    def get_label(self):
        """ convert name to label,
            and optionally permutate some labels
        """
        names = {item['name'] for item in self.data_items}
        names = sorted(list(names))
        self.classes = names
        name2label = {name: idx for idx, name in enumerate(names)}

        self.label_items = []
        for item in self.data_items:
            label = name2label[item['name']]
            self.label_items.append({'label': label})

        if self.noise_ratio:
            self.corrupt_label()

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        # load image and pre-process (pipeline)
        path = self.data_items[idx]['path']
        item = {'path': osp.join(self.data_dir, path)}
        
        path = item['path']
        #image = Image.open(path).convert('RGB')
        image = cv2.imread(path)
        if image is None:
            raise OSError('{} is not found'.format(path))
        elif image.shape[2] == 1:
            # If the image has 1 channel, it's likely grayscale, so convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = np.array(image)
        image = image[:, :, ::-1]

        # align the face image if the source and target landmarks are given
        src_landmark = item.get('src_landmark')
        tgz_landmark = item.get('tgz_landmark')
        crop_size = item.get('crop_size')
        if not (src_landmark is None or tgz_landmark is None or crop_size is None):
            tform = transform.SimilarityTransform()
            tform.estimate(tgz_landmark, src_landmark)
            M = tform.params[0:2, :]
            image = cv2.warpAffine(image, M, crop_size, borderValue=0.0)

        # normalize to [-1, 1]
        image = ((image - 127.5) / 127.5)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        if not self.test_mode and random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
        #********************#
        
        label = self.label_items[idx]['label']

        return image, label