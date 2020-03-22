import torch
from torchvision import transforms, datasets
from our_parser import get_config
import os
config = get_config()
# train_dir = os.path.join(root_dir, 'GestNet/train/')
# train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
# train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

# val_dir = cfg.VAL_DATASET_DIR
# val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
# val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=2)
import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split


class ImageFolderSplitter:
    '''images should be placed in folders like:
    --root
    ----root\dogs
    ----root\dogs\image1.png
    ----root\dogs\image2.png
    ----root\cats
    ----root\cats\image1.png
    ----root\cats\image2.png
    path: the root of the image folder'''
    def __init__(self, path, train_size=0.9):
        self.path = path
        self.train_size = train_size
        self.x_train = []
        self.x_valid = []
        self.y_train = []
        self.y_valid = []
        data = h5py.File(path, "r")
        x_data = np.array(data['X'])  # h5py._hl.dataset.Dataset -> numpy array
        y_data = np.array(data['Y'])
        # print(type(X_data))
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x_data, y_data, train_size=0.9, test_size=0.1, random_state=22)
        print(self.x_train.shape)
        print(self.y_train.shape)

    @staticmethod
    def to_categorical(y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    def getTrainingDataset(self):
        return self.x_train, self.y_train

    def getValidationDataset(self):
        return self.x_test, self.y_test


class DatasetFromFilename(Dataset):
    # x: a list of image file full path
    # y: a list of image categories
    def __init__(self, x, y, transforms=None):
        super(DatasetFromFilename, self).__init__()
        self.x = x
        self.y = y
        if transforms is None:
            self.transforms = transforms.ToTensor()
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # print(self.x[idx].shape)
        img = Image.fromarray(self.x[idx].astype(np.uint8))
        # img = Image.open(img)
        # img = img.convert("RGB")
        # print(self.y.shape)
        return self.transforms(img), torch.tensor(self.y[idx])


# test code
# mode = config['mode']
# assert mode in ['train', 'test']
# train_dir = '%s/%s/%s' % (config['root_dir'], 'GestNet', mode)


def in_test():
    batch_size = config['batch_size']

    train_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    val_transforms = transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    store_dir = '%s/%s' % (config['root_dir'], 'dataset/data.h5')
    assert os.path.exists(store_dir)
    splitter = ImageFolderSplitter(store_dir)

    x_train, y_train = splitter.getTrainingDataset()
    training_dataset = DatasetFromFilename(x_train, y_train, transforms=train_transforms)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    x_valid, y_valid = splitter.getValidationDataset()
    validation_dataset = DatasetFromFilename(x_valid, y_valid, transforms=val_transforms)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    for t, (x, y) in enumerate(training_dataloader):
        if t % 100 == 0:
            # this converts it from GPU to CPU and selects first image
            img = x.cpu().numpy().astype(np.uint8)[0]
            # convert image back to Height,Width,Channels
            img = np.transpose(img, (1, 2, 0))
            # print(x.numpy().astype(np.uint8)[0].reshape([64, 64, 3]).shape)
            img = Image.fromarray(img)
            # img.show()
            # print(y)
            print(x.shape, y.shape)


if __name__ == '__main__':
    in_test()
