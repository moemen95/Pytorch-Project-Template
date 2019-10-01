import gin
import torch.optim as optim


@gin.configurable(blacklist=['params'])
def sgd(learning_rate, momentum, params):
    optimizer = optim.SGD(
        params=params,
        lr=learning_rate,
        momentum=momentum,
    )

    return optimizer
