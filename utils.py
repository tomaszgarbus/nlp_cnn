import os
import numpy as np
from typing import List, Dict


def listfiles(path: str) -> List[str]:
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def nearest_neighbor(embedding: np.ndarray, glove: Dict[str, np.ndarray]) -> str:
    if np.linalg.norm(embedding) == 0.:
        # Most likely an unknown or empty word
        return ""
    cur_min = 1000000
    ret = ""
    for key in glove:
        tmp = np.linalg.norm(glove[key] - embedding)
        if tmp < cur_min:
            cur_min = tmp
            ret = key
    return ret