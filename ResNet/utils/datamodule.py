from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets import ImageFolder

import os
import sys
sys.path.append('.')

from utils.ImageFolderSplit import ImageFolderSplitter, DatasetFromFilename

class TinyImagenetDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, train_transforms, val_transforms, test_transforms, train_size=0.9, num_workers=16):
        super().__init__()
        self.data_dir=data_dir
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers
    
    def setup(self, stage):
        if stage in (None, 'fit'):
            # splitter = ImageFolderSplitter(path=os.path.join(self.data_dir,'train'), train_size=self.train_size)
            # X_train, y_train = splitter.getTrainingDataset()
            # self.training_dataset = DatasetFromFilename(X_train, y_train, transforms=self.train_transforms)
            # X_valid, y_valid = splitter.getValidationDataset()
            # self.validation_dataset = DatasetFromFilename(X_valid, y_valid, transforms=self.val_transforms)
            train_dataset = ImageFolder(root=os.path.join(self.data_dir, 'train'), transform=self.train_transforms)
            train_size = int(self.train_size * len(train_dataset))
            val_size = len(train_dataset) - train_size
            self.training_dataset, self.validation_dataset = random_split(train_dataset, [train_size, val_size])
        if stage in (None, 'test'):
            self.test_dataset = ImageFolder(root=os.path.join(self.data_dir, 'val'), transform=self.test_transforms)
    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=5, shuffle=False, num_workers=self.num_workers)
