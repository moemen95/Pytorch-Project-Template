## How to Start your PyTorch from Scratch:

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
Dataloader is responsible for your dataset loading utils and returning a PyTorch dataloader for both training and validation sets.

#### Model:
#### Config:
### 3. Identify your changes:
If this is one of the categories we covered, you may reuse part of the codes in most of the files.
If your problem was not introduced before in the template, you will use the example.py files for your reference.

Before editing any file, you need to identify the changes you need to make to any of the blocks mentioned above.
- **Dataloader**: If you are using any of the datasets mentioned in the template, use the same dataloader; if not, use the same logic and change whenever necessary.
- **Model**: define your model by writing the model init and forward function, using the same logic we used here. If you need to define any custom layers or blocks, add them in the custom_layers folder.
- **Config**: duplicate the given example for the config file with a new name and adapt its values; add or remove config fields whenever necessary.
- **Agent**: duplicate the given agent example and start defining the main functions as needed. Feel free to reuse any of the codes given in the examples.

### 4. Embed your new codes:
### 5. Run your desired configs:
The run file runs the main file given the config file name. You need to change nothing here but the config file name as an argument when running.