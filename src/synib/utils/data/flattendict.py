import torch

def flatten_loss_dict(loss_dict: dict, parent_key: str = "", sep: str = "_") -> dict:
    """
    Recursively flattens a nested dictionary of loss tensors.

    Example:
        Input:
            {
                "px1": {"rec": t1, "kl": t2},
                "px2": {"rec": t3, "kl": t4},
                "sl": t5,
                "infonce": t6
            }
        Output:
            {
                "px1_rec": t1,
                "px1_kl": t2,
                "px2_rec": t3,
                "px2_kl": t4,
                "sl": t5,
                "infonce": t6
            }

    Args:
        loss_dict: potentially nested dictionary containing tensors.
        parent_key: prefix to prepend (used for recursion).
        sep: separator used between parent and child keys.

    Returns:
        A flat dictionary mapping concatenated keys to tensors.
    """
    flat_dict = {}
    for key, value in loss_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            nested = flatten_loss_dict(value, parent_key=new_key, sep=sep)
            flat_dict.update(nested)
        elif torch.is_tensor(value):
            flat_dict[new_key] = value

    return flat_dict