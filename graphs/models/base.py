import torch.nn as nn


class BaseModel(nn.Module):
    @property
    def name(self):
        return self.__class__.__name__
