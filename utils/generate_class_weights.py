import numpy as np

from sklearn.utils.class_weight import compute_class_weight

import os

import numpy as np
import scipy.io as sio
import PIL
from PIL import Image
import torch

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as standard_transforms

import utils.voc_utils as extended_transforms
from utils.voc_utils import make_dataset


class VOC(data.Dataset):
    def __init__(self, mode, data_root, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode, data_root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'train':
            mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            mask = Image.open(mask_path)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)


class VOCDataLoader:
    def __init__(self, config):
        self.config = config
        assert self.config.mode in ['train', 'test', 'random']

        mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

        self.input_transform = standard_transforms.Compose([
            standard_transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
            extended_transforms.FlipChannels(),
            standard_transforms.ToTensor(),
            standard_transforms.Lambda(lambda x: x.mul_(255)),
            standard_transforms.Normalize(*mean_std)
        ])

        self.target_transform = standard_transforms.Compose([
            standard_transforms.Resize((256, 256), interpolation=PIL.Image.NEAREST),
            extended_transforms.MaskToTensor()
        ])

        self.restore_transform = standard_transforms.Compose([
            extended_transforms.DeNormalize(*mean_std),
            standard_transforms.Lambda(lambda x: x.div_(255)),
            standard_transforms.ToPILImage(),
            extended_transforms.FlipChannels()
        ])

        self.visualize = standard_transforms.Compose([
            standard_transforms.Resize(400),
            standard_transforms.CenterCrop(400),
            standard_transforms.ToTensor()
        ])
        if self.config.mode == 'random':
            train_data = torch.randn(self.config.batch_size, self.config.input_channels, self.config.img_size,
                                     self.config.img_size)
            train_labels = torch.ones(self.config.batch_size, self.config.img_size, self.config.img_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        elif self.config.mode == 'train':
            train_set = VOC('train', self.config.data_root,
                            transform=self.input_transform, target_transform=self.target_transform)
            valid_set = VOC('val', self.config.data_root,
                            transform=self.input_transform, target_transform=self.target_transform)

            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
            self.valid_iterations = (len(valid_set) + self.config.batch_size) // self.config.batch_size

        elif self.config.mode == 'test':
            test_set = VOC('test', self.config.data_root,
                           transform=self.input_transform, target_transform=self.target_transform)

            self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False,
                                          num_workers=self.config.data_loader_workers,
                                          pin_memory=self.config.pin_memory)
            self.test_iterations = (len(test_set) + self.config.batch_size) // self.config.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass


def calculate_weigths_labels():
    class Config:
        mode = "train"
        num_classes = 21
        batch_size = 32
        max_epoch = 150

        validate_every = 2

        checkpoint_file = "checkpoint.pth.tar"

        data_loader = "VOCDataLoader"
        data_root = "../data/pascal_voc_seg/"
        data_loader_workers = 4
        pin_memory = True
        async_loading = True

    # Create an instance from the data loader
    from tqdm import tqdm
    data_loader = VOCDataLoader(Config)
    z = np.zeros((Config.num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(data_loader.train_loader, total=data_loader.train_iterations)

    for _, y in tqdm_batch:
        labels = y.numpy().astype(np.uint8).ravel().tolist()
        z += np.bincount(labels, minlength=Config.num_classes)
    tqdm_batch.close()
    # ret = compute_class_weight(class_weight='balanced', classes=np.arange(21), y=np.asarray(labels, dtype=np.uint8))
    total_frequency = np.sum(z)
    print(z)
    print(total_frequency)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    np.save('../pretrained_weights/voc2012_256_class_weights', ret)
    print(ret)


if __name__ == '__main__':
    calculate_weigths_labels()
