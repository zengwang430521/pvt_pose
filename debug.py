from models.pvt import pvt_tiny, pvt_small, pvt2048_small
from models.hmr import HMR
import torch

# model = pvt_small()
# model = HMR()
model = pvt2048_small()
img = torch.zeros([1, 3, 224, 224])
out = model(img)
t = 0
