import torch.nn as nn


def get_loss_function(loss_type: str = "cross_entropy_loss"):

    if loss_type == "cross_entropy_loss":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplmentedError(f"loss type: {loss_type} not implemented")
