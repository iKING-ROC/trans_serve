from model.astgcn import ASTGCN
import sys
import numpy as np

sys.path.append('.')
from model.model_config import get_backbones
import torch
device = torch.device('cpu')
all_backbones = get_backbones('configurations/PEMS04.conf',
                             'data/PEMS04/distance.csv', device)

net = ASTGCN(12, all_backbones, 307, 3, [[24, 12], [12, 12], [24, 12]], device)

test_w = torch.randn(16, 307, 3, 24).to(device)
test_d = torch.randn(16, 307, 3, 12).to(device)
test_r = torch.randn(16, 307, 3, 24).to(device)
output = net([test_w, test_d, test_r])
print(output)
assert output.shape == (16, 307, 12)
assert type(output.detach().numpy().mean()) == np.float32
