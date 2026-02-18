import torch
import random
import numpy as np
from accelerate.utils import set_seed

def deterministic(seed):
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    # set_seed(seed, device_specific=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
