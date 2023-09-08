import os.path as osp
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import cv2

from skimage import transform

from .utils import get_metrics


class PairDataset(Dataset):
    def __init__(self, name, data_dir, ann_path, metrics,
            test_mode=True):
        super().__init__()

        self.name = name
        self.data_dir = data_dir
        self.ann_path = ann_path
        self.metrics = metrics
        self.test_mode = test_mode

        self.get_data()
        self.get_label()

    def get_data(self):
        """Get data from an annotation file.
        """
        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

        paths = set()
        for line in lines:
            _, path1, path2 = line.rstrip().split(' ')
            paths.add(path1)
            paths.add(path2)
        paths = list(paths)
        paths.sort()
        self.data_items = [{'path': path} for path in paths]

        if len(self.data_items) == 0:
            raise (RuntimeError('Found 0 files.'))

    def get_label(self):
        """Get labels from an annoation file
        """
        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

        path2index = {item['path']: idx 
                for idx, item in enumerate(self.data_items)}

        self.indices0 = []
        self.indices1 = []
        self.labels = []
        for line in lines:
            label, path0, path1 = line.rstrip().split(' ')
            self.indices0.append(path2index[path0])
            self.indices1.append(path2index[path1])
            self.labels.append(int(label))

    def evaluate(self, feats, 
            FPRs=['1e-4', '5e-4', '1e-3', '5e-3', '5e-2']):
        # pair-wise scores
        feats = F.normalize(feats, dim=1)
        feats0 = feats[self.indices0, :]
        feats1 = feats[self.indices1, :]
        scores = torch.sum(feats0 * feats1, dim=1).tolist()

        return get_metrics(self.labels, scores, FPRs)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        # load image and pre-process (pipeline) from path
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

        return image, idx
