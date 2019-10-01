"""
Mnist Data loader, as given in Mnist tutorial
"""
import gin
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


@gin.configurable
class MnistDataLoader:
    def __init__(
            self,
            data_mode,
            batch_size,
            test_batch_size,
            pin_memory,
            num_workers,
            rand_input_channels=None,
            rand_img_size=None,
    ):
        if data_mode == "download":
            self.train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=batch_size, shuffle=True, num_workers=num_workers,
                pin_memory=pin_memory)
            self.val_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
                batch_size=test_batch_size, shuffle=True, num_workers=num_workers,
                pin_memory=pin_memory)
        elif data_mode == "imgs":
            raise NotImplementedError("This mode is not implemented YET")

        elif data_mode == "numpy":
            raise NotImplementedError("This mode is not implemented YET")

        elif data_mode == "random":
            assert None not in (rand_img_size,
                                rand_input_channels), 'Specify rand_img_size and rand_input_channels with data_mode=="random"'
            train_data = torch.randn(batch_size, rand_input_channels, rand_img_size, rand_img_size)
            train_labels = torch.ones(batch_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + batch_size - 1) // batch_size
            self.valid_iterations = (self.len_valid_data + batch_size - 1) // batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def finalize(self):
        pass
