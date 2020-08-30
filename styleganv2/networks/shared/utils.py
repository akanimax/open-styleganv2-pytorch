import numpy as np
from torch.nn import Module


def nf(
    stage: int, fmap_base: int, fmap_decay: float, fmap_min: int, fmap_max: int
) -> int:
    """
    computes the number of fmaps present in each stage
    Args:
        stage: stage level
        fmap_base: base number of fmaps
        fmap_decay: decay rate for the fmaps in the network
        fmap_min: minimum number of fmaps
        fmap_max: maximum number of fmaps
    Returns: number of fmaps that should be present there
    """
    return int(
        np.clip(
            int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max
        ).item()
    )


def update_average(model_tgt: Module, model_src: Module, beta: float) -> None:
    """
    function to calculate the Exponential moving averages for the model weights.
    This function updates the exponential average weights based on the current training
    Args:
        model_tgt: target model
        model_src: source model
        beta: value of decay beta
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_tgt
        (p_tgt.mul_(beta)).add_((1.0 - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)
