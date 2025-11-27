# survival_v2/src/utils/shap_utils_v2.py

import torch

def parse_eid(x):
    """
    Safely parse EID from a variety of possible representations:
    - int
    - float
    - numpy scalar
    - Python string "5238560"
    - PyTorch tensor(5238560)
    - String representation "tensor(5238560)"
    """

    # Case 1: PyTorch zero-dimensional tensor
    if isinstance(x, torch.Tensor):
        return int(x.item())

    # Case 2: String like "tensor(5238560)"
    if isinstance(x, str) and x.startswith("tensor(") and x.endswith(")"):
        # Strip "tensor(" and ")"
        try:
            return int(x.replace("tensor(", "").replace(")", ""))
        except:
            pass

    # Case 3: Normal string or number
    try:
        return int(x)
    except:
        raise ValueError(f"Cannot parse EID value: {x}")
