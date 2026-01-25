
import random, torch, numpy as np
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
