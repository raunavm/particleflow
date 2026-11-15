import torch
from mlpf.optimizers.lamb import Lamb
from mlpf.logger import _logger
from mlpf.conf import MLPFConfig, OptimizerType


def get_optimizer(model: torch.nn.Module, config: MLPFConfig):
    """
    Returns the optimizer for the given model based on the configuration provided.
    Parameters:
    model (torch.nn.Module): The model for which the optimizer is to be created.
    config (MLPFConfig): Configuration object.
    Returns:
    torch.optim.Optimizer: The optimizer specified in the configuration.
    Raises:
    ValueError: If the specified optimizer type is not supported.
    """

    wd = config.weight_decay
    parameters = model.parameters()

    if config.optimizer == OptimizerType.ADAMW:
        ret = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=wd)
    elif config.optimizer == OptimizerType.LAMB:
        # gradient_clip_norm is LAMB's own global-norm clip, whose default is 1.0
        clip = {} if config.gradient_clip_norm is None else {"max_grad_norm": config.gradient_clip_norm}
        ret = Lamb(parameters, lr=config.lr, weight_decay=wd, **clip)
    elif config.optimizer == OptimizerType.SGD:
        ret = torch.optim.SGD(parameters, lr=config.lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer}")

    _logger.info(f"Created optimizer: {ret}")
    return ret
