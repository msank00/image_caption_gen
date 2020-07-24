import torch
import torch.nn as nn


def get_optimizer(
    params: list, optim_type: str = "adam", learning_rate: float = 0.001
):

    if optim_type == "adam":
        return torch.optim.Adam(params=params, lr=0.001)
    else:
        raise NotImplmentedError(
            f"Optimizer type: {optim_type} not implemented"
        )
