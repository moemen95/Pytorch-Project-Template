## Getting Started:

Here we will walk through the process of starting a new PyTorch project using our template.
#### 1. Main Building blocks:
Let's start by reviewing the folder structure of the project.
```
├── agents
|  └── example.py
├── configs
|  └── example_exp_0.py
├── data
├── datasets
|  └── example.py
├── experiments
├── graphs
|  └── models
|  |  └── custome_layers
|  |  |  └── example.py
|  |  └── example.py
|  └── losses
|  |  └── example.py
├── pretrained_weights
├── tutorials
├── utils
|  └── assets
├── main.py
└── run.sh
```

#### Agent:
The agent controls the training and evaluation process of your model and is considered the core of the project.
We are providing a base agent to inherit from; it includes the following functions that should be overridden in your custom agent:
- These functions are responsible for checkpoint loading and saving:
```python
def load_checkpoint(self, file_name):
    """
    Latest checkpoint loader
    :param file_name: name of the checkpoint file
    :return:
    """

def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
    """
    Checkpoint saver
    :param file_name: name of the checkpoint file
    :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
    :return:
    """
```
- These functions are responsible for training and validation:

```python
def run(self):
    """
    The main operator
    :return:
    """
    
def train(self):
    """
    Main training loop
    :return:
    """

def train_one_epoch(self):
    """
    One epoch of training
    :return:
    """

def validate(self):
    """
    One cycle of model validation
    :return:
    """

def finalize(self):
    """
    Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
    :return:
    """
```

#### Dataloader:
Dataloader is responsible for your dataset utilities and returns a PyTorch dataloader for both training and validation splits. You can specify the data loading mode in the config as numpy, imgs, etc, as seen below:
```json
    if config.data_mode == "numpy_train":
            normalize = v_transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
                train_set = v_datasets.CIFAR10('./data', train=True, download=True,
                                         transform=v_transforms.Compose([
                                             v_transforms.RandomCrop(32, padding=4),
                                             v_transforms.RandomHorizontalFlip(),
                                             v_transforms.ToTensor(),
                                             normalize,
                                         ]))
            valid_set = v_datasets.CIFAR10('./data', train=False,
                                       transform=v_transforms.Compose([
                                           v_transforms.ToTensor(),
                                           normalize,
                                       ]))
    
        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False)
    
        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)
```
#### Model:
This is where you define your main model, as shown in the example below:
 
```Python
class ModelExample(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # define layers
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(in_channels=self.config.input_channels, out_channels=self.config.num_filters, kernel_size=3, stride=1, padding=1, bias=False)

        # initialize weights
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        out = x.view(x.size(0), -1)
        return out
```

#### Config:
The usage of a config file is what keeps your code dynamic. This will include all the hyper-parameters that control your data, all training variables and model saving.

Should you have any new parameters that you think can be added to the configurations file, feel free to directly embed them into the configurations to be used within your project, as found below.

```json
{
  "newKey" : "newValue"
}
```
This can be accessed inside the code as 
```python 
parameter_value = config.newkey
```

### 2. Identify your changes:
If this is one of the categories we covered, you may reuse part of the codes in most of the files.
If your problem was not introduced before in the template, you will use the ```example.py``` files for your reference.

Before editing any file, you need to identify the changes you need to make to any of the blocks mentioned above.
- **Agent**: 
    Duplicate the given agent example and start defining the main functions as needed. Feel free to reuse any of the codes given in the examples.
- **Dataloader**: 
    If you are using any of the datasets mentioned in the template, use the same dataloader; if not, use the same logic and change whenever necessary.
- **Model**: 
    Define your model by writing the model init and forward function, using the same logic we used here. If you need to define any custom layers or blocks, add them in the custom_layers folder.
- **Config**: 
    Duplicate the given example for the config file with a new name and adapt its values; add or remove config fields whenever necessary.

### 3. Run your desired configs:
After you make your changes to the main blocks, you will find a bash script that runs the main file. You need to change nothing here but the config file name as an argument upon running. 

This will enable you to have more than one model with all its variations inside the same project, where you can run any from the same file given different configs.