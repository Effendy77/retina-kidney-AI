import torch
import numpy as np
import re


def parse_eid(x):
    """
    Safely parse an EID (UK Biobank participant ID) from multiple 
    possible types or string representations.

    Handles:
    - Python int or float
    - numpy integer / float scalar (np.int64, np.float64)
    - PyTorch scalar Tensor (torch.tensor(5238560))
    - string: "5238560"
    - string: "  5238560  "
    - string: "tensor(5238560)"
    
    Ensures:
    - returned type: Python int
    - rejects invalid types with clear error messages
    """

    # ==========================================================
    # 1) PyTorch Scalar Tensor
    # ==========================================================
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:
            return int(x.item())
        raise ValueError(f"Tensor must be 0-dimensional, got shape {tuple(x.shape)}")

    # ==========================================================
    # 2) Numpy scalar values (np.int64, np.float64, etc.)
    # ==========================================================
    if isinstance(x, np.generic):
        try:
            return int(x)
        except Exception:
            raise ValueError(f"Cannot convert numpy scalar to int: {x}")

    # ==========================================================
    # 3) Clean strings representing tensor(...) such as:
    #       "tensor(5238560)"
    #       " tensor(5238560) "
    # ==========================================================
    if isinstance(x, str):
        s = x.strip()

        # Case 3A: string starts with tensor(###)
        if s.startswith("tensor(") and s.endswith(")"):
            inner = s[len("tensor("):-1]
            if inner.isdigit():
                return int(inner)
            # Try regex extraction
            match = re.search(r"\d+", inner)
            if match:
                return int(match.group(0))
            raise ValueError(f"Cannot parse tensor-style EID string: {x}")

        # Case 3B: plain string "5238560"
        # Remove whitespace and check digits
        s_clean = re.sub(r"[^\d]", "", s)  # keep only digits
        if s_clean.isdigit():
            return int(s_clean)

        raise ValueError(f"Cannot parse EID string: {x}")

    # ==========================================================
    # 4) Regular numeric values
    # ==========================================================
    try:
        return int(x)
    except Exception:
        raise ValueError(f"Cannot parse EID value: {x}")
