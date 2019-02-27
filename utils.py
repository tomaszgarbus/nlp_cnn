import os
import numpy as np
from typing import List, Dict


def listfiles(path: str) -> List[str]:
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]